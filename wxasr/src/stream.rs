//! 伪流式识别。
//!
//! 模型注意力非因果，没有 KV cache，无法增量解码；流式做法是分块
//! 重算：每块带前后文上下文解码一次，定稿后的前缀不再变化。参数
//! 取自官方 `sr_online.conf` 的 `block_spec = 100:400:100`（10ms
//! 特征帧）：前文 100 帧 / 块 400 帧 / 后文 100 帧。

use crate::fbank::{Fbank, DIM};
use crate::model::Model;
use std::collections::HashSet;
use std::sync::Arc;

/// 解码窗口中、定稿块之前的前文帧数
pub const PREV: usize = 100;
/// 每次定稿的块长（特征帧）
pub const BLK: usize = 400;
/// 定稿块之后的后文帧数
pub const POST: usize = 100;
/// 每个解码窗口末尾补的静音帧数：窗口在语音中途截断时，模型常整窗
/// 不出字，补静音后稳定
pub const PAD: usize = 50;
/// 结束时整段一次解码的长度上限（特征帧）：20 秒内定稿等待约半秒，
/// 70 秒要近 4 秒，超过后改用已定稿块 + 尾部
pub const WHOLE_MAX: usize = 2000;
/// 轮换时判定「真静音收尾」所需的末尾连续 blank 输出数（每个 50ms）
const SILENT_TAIL: usize = 8;

/// 伪流式识别器：喂 PCM，边说边出中间结果，`finish` 出定稿。
pub struct Stream {
    model: Arc<Model>,
    fbank: Fbank,
    frames: Vec<[f32; DIM]>,
    /// 已定稿的 token id（CTC 折叠前）
    committed: Vec<u32>,
    /// 已定稿覆盖到的特征帧位置
    cursor: usize,
    /// 接续携带的左文帧数（见 [`Stream::rotate`]），自有内容从该位置起
    lead: usize,
    fresh: usize,
    /// 词表中标点 token 的 id
    punct: HashSet<u32>,
}

impl Stream {
    pub fn new(model: Arc<Model>) -> Self {
        let punct = model
            .vocab()
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                let mut c = t.chars();
                matches!((c.next(), c.next()), (Some(ch), None) if PUNCT.contains(&ch))
            })
            .map(|(i, _)| i as u32)
            .collect();
        Self {
            model,
            fbank: Fbank::new(),
            frames: Vec::new(),
            committed: Vec::new(),
            cursor: 0,
            lead: 0,
            fresh: 0,
            punct,
        }
    }

    /// 喂入一段 PCM（16kHz 单声道 i16）
    pub fn push(&mut self, pcm: &[i16]) {
        let n = self.frames.len();
        self.fbank.push(pcm, &mut self.frames);
        self.fresh += self.frames.len() - n;
        while self.frames.len() >= self.cursor + BLK + POST {
            self.commit();
        }
    }

    /// 距上次 [`partial`](Self::partial) 新增的特征帧数，调用方可据此节流
    pub fn fresh(&self) -> usize {
        self.fresh
    }

    /// 解码 `[cursor - prev, end)` 并在尾部补 `pad` 帧静音，
    /// 返回 cursor 之后的输出帧 id
    fn decode(&self, prev: usize, end: usize, pad: usize) -> Vec<u32> {
        let a = self.cursor.saturating_sub(prev);
        let mut x: Vec<[f32; DIM]> = self.frames[a..end].to_vec();
        if pad > 0 {
            x.resize(x.len() + pad, self.fbank.silence());
        }
        let ids = self.model.forward(&x);
        ids.get((self.cursor - a) / 5..).map(|s| s.to_vec()).unwrap_or_default()
    }

    /// 定稿一块：解码后截到块尾附近的连续 blank 处，避免字恰好落在
    /// 块边界上被两边都丢掉
    fn commit(&mut self) {
        let mut ids = self.decode(PREV, self.cursor + BLK + POST, PAD);
        ids.truncate(BLK / 5);
        // 在块后半段找最后一对连续 blank，在其前截断
        if let Some(c) = (ids.len() / 2..ids.len())
            .rev()
            .find(|&c| ids[c] == 0 && ids[c - 1] == 0)
        {
            ids.truncate(c);
        }
        self.cursor += ids.len() * 5;
        self.committed.extend(ids);
    }

    /// 中间结果：已定稿部分 + 尾部临时解码（前文减半省算力）。
    /// 去掉末尾标点，避免边说边上屏时句号反复增删。
    /// 每次调用都会触发一次解码，调用方应按 [`fresh`](Self::fresh) 节流。
    pub fn partial(&mut self) -> String {
        self.fresh = 0;
        let mut ids = self.committed.clone();
        if self.frames.len() > self.cursor {
            ids.extend(self.decode(PREV / 2, self.frames.len(), PAD));
        }
        tidy(&self.model.text(&ids))
            .trim_end_matches(PUNCT)
            .to_string()
    }

    /// 结束识别，返回定稿文本。
    /// 短音频（自有内容 ≤20s）整段一次解码更准（携带的左文参与注意力
    /// 但不计入输出）；超长音频用已定稿块 + 尾部。
    pub fn finish(mut self) -> String {
        let ids = self.decode_all(0);
        tidy(&self.model.text(&ids))
    }

    /// 定稿当前流并返回接续流，用于长会话的资源轮换：旧流的帧缓冲与
    /// 定稿 token 随返回的文本交给调用方后丢弃，新流只带走：
    ///
    /// - 切点前 `PREV` 帧左文与特征器状态（滑动均值、未成帧样本）——
    ///   切点后模型看到的输入与不切割时一致，且左文不重复出字
    ///   （decode 的输出截取天然跳过 `cursor` 之前的帧）
    /// - 切点之后尚未定稿的帧——切点取在后半段最后一对连续 blank 之前
    ///   （同 [`commit`](Self::commit)），避免说话中轮换时跨切点的字
    ///   两边都出不来；这些帧由新流带着完整右文重新解码
    ///
    /// 返回的文本只覆盖到切点；静音处轮换时切点即末尾。
    pub fn rotate(self) -> (String, Stream) {
        self.rotate_at(false)
    }

    /// 同 [`rotate`](Self::rotate)，但切点优先取最后一个**已定稿**标点
    /// 之后——返回的文本以标点收尾，读起来是完整的句/分句。标点只在
    /// 已定稿区找（那里的 id 不会被后续解码改写）；找不到时退回
    /// `rotate` 的 blank 切点。
    pub fn rotate_at_punct(self) -> (String, Stream) {
        self.rotate_at(true)
    }

    /// 最近一个已定稿标点的位置（毫秒，自本流自有内容起算）。只看
    /// 分块定稿的 id，结果稳定、不随中间结果回改；代价是最多滞后一个
    /// 定稿块（约 5s）。
    pub fn punct_ms(&self) -> Option<u32> {
        self.committed
            .iter()
            .rposition(|id| self.punct.contains(id))
            .map(|i| i as u32 * 50) // 每个 id 对应 5 帧 = 50ms
    }

    fn rotate_at(mut self, punct: bool) -> (String, Stream) {
        let mut ids = self.decode_all(PAD);
        // 补静音区的输出不对应真实帧，但句末标点常落在这里，先记下
        let real = ((self.frames.len() - self.lead) / 5).min(ids.len());
        let pad_punct = ids[real..].iter().copied().find(|id| self.punct.contains(id));
        ids.truncate(real);
        let stable = ((self.cursor - self.lead) / 5).min(ids.len());
        // 末尾连续 blank 够长才算真静音收尾：说话中截断时未说完的字也
        // 表现为 blank，但只有几个（不足一个音节）
        let silent_tail = ids.iter().rev().take_while(|id| **id == 0).count() >= SILENT_TAIL;
        let cut = punct
            .then(|| ids[..stable].iter().rposition(|id| self.punct.contains(id)))
            .flatten()
            .map(|p| p + 1)
            .or_else(|| {
                if silent_tail {
                    return None; // 整段定稿，句末标点不丢给下一段
                }
                ((ids.len() / 2).max(1)..ids.len())
                    .rev()
                    .find(|&c| ids[c] == 0 && ids[c - 1] == 0)
            });
        match cut {
            Some(c) => ids.truncate(c),
            // 真静音收尾时补静音与实际一致，其句末标点可信
            None if silent_tail => ids.extend(pad_punct),
            None => {}
        }
        let cut = self.lead + ids.len() * 5;
        let start = cut.saturating_sub(PREV);
        let frames = self.frames.split_off(start);
        let lead = cut - start;
        (
            tidy(&self.model.text(&ids)),
            Stream {
                model: self.model,
                fbank: self.fbank,
                cursor: lead,
                lead,
                fresh: frames.len() - lead,
                frames,
                committed: Vec::new(),
                punct: self.punct,
            },
        )
    }

    /// 自有内容（`lead` 之后）的全部输出 id，按 5 帧一个与 `frames[lead..]`
    /// 对齐；`pad` > 0 时尾部补静音（输出里多出 `pad / 5` 个）
    fn decode_all(&mut self, pad: usize) -> Vec<u32> {
        if self.frames.len() - self.lead <= WHOLE_MAX {
            let ids = if pad > 0 {
                let mut x = self.frames.clone();
                x.resize(x.len() + pad, self.fbank.silence());
                self.model.forward(&x)
            } else {
                self.model.forward(&self.frames)
            };
            return ids.get(self.lead / 5..).map(|s| s.to_vec()).unwrap_or_default();
        }
        let mut ids = std::mem::take(&mut self.committed);
        if self.frames.len() > self.cursor {
            ids.extend(self.decode(PREV, self.frames.len(), PAD));
        }
        ids
    }
}

const PUNCT: &[char] = &['，', '。', '？', '！', '、', ',', '.', '?', '!'];

/// 开口前的底噪偶尔被识别成句首逗号；偶尔出现「，。」这样的连续
/// 标点——只留后一个
pub(crate) fn tidy(t: &str) -> String {
    let chars: Vec<char> = t.trim_start_matches(PUNCT).chars().collect();
    chars
        .iter()
        .enumerate()
        .filter(|&(i, &c)| {
            !(PUNCT.contains(&c) && chars.get(i + 1).is_some_and(|n| PUNCT.contains(n)))
        })
        .map(|(_, &c)| c)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tidy_strips_and_collapses_punctuation() {
        assert_eq!(tidy("，你好。"), "你好。");
        assert_eq!(tidy("你好，。世界"), "你好。世界");
        assert_eq!(tidy("你好，"), "你好，");
        assert_eq!(tidy("没有标点"), "没有标点");
    }
}

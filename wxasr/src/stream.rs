//! 伪流式识别。
//!
//! 模型注意力非因果，没有 KV cache，无法增量解码；流式做法是分块
//! 重算：每块带前后文上下文解码一次，定稿后的前缀不再变化。参数
//! 取自官方 `sr_online.conf` 的 `block_spec = 100:400:100`（10ms
//! 特征帧）：前文 100 帧 / 块 400 帧 / 后文 100 帧。

use crate::fbank::{Fbank, DIM};
use crate::model::Model;
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

/// 伪流式识别器：喂 PCM，边说边出中间结果，`finish` 出定稿。
pub struct Stream {
    model: Arc<Model>,
    fbank: Fbank,
    frames: Vec<[f32; DIM]>,
    /// 已定稿的 token id（CTC 折叠前）
    committed: Vec<u32>,
    /// 已定稿覆盖到的特征帧位置
    cursor: usize,
    fresh: usize,
}

impl Stream {
    pub fn new(model: Arc<Model>) -> Self {
        Self {
            model,
            fbank: Fbank::new(),
            frames: Vec::new(),
            committed: Vec::new(),
            cursor: 0,
            fresh: 0,
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
    /// 短音频（≤20s）整段一次解码更准；超长音频用已定稿块 + 尾部。
    pub fn finish(mut self) -> String {
        if self.frames.len() <= WHOLE_MAX {
            return tidy(&self.model.text(&self.model.forward(&self.frames)));
        }
        if self.frames.len() > self.cursor {
            let tail = self.decode(PREV, self.frames.len(), PAD);
            self.committed.extend(tail);
        }
        tidy(&self.model.text(&self.committed))
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

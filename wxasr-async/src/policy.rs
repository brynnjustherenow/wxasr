//! 上下文轮换策略：何时定稿当前流并重建，约束长会话下 [`wxasr::Stream`]
//! 内部状态的线性增长（特征帧缓冲约 16KB/s，已定稿 token 与中间结果
//! 计算同样随总长线性涨）。每次轮换即一次定稿，产生一个
//! [`Final`](crate::AsrEvent::Final)；此后的
//! [`Partial`](crate::AsrEvent::Partial) 从空开始。

use std::fmt;
use std::sync::Arc;

/// `Auto` 策略的静音门：软上限到达后，需连续静音超过这么多毫秒才轮换
/// （换到停顿处切割，避免词中途被截断）
const SILENCE_GATE_MS: u32 = 500;

/// 轮换决策依据（当前音频流的状态快照）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotateCtx {
    /// 自上次轮换以来喂入的音频毫秒数（不含接续携带的未定稿尾部）
    pub stream_ms: u32,
    /// 音频尾部当前的连续静音毫秒数（描述音频本身，轮换不清零）
    pub silence_ms: u32,
    /// 最近一个已定稿标点的位置（毫秒，自本流起算）；无则 `None`。
    /// 取自分块定稿的 token，稳定不回改，最多滞后一个定稿块（约 5s）
    pub punct_ms: Option<u32>,
}

/// 轮换策略。`Default` 为 [`RotatePolicy::Auto`] { soft_ms: 15000, hard_ms: 20000 }。
#[derive(Clone)]
pub enum RotatePolicy {
    /// 双上限：累计 `soft_ms` 后于下一次 ≥500ms 静音处轮换；累计 `hard_ms` 即使说话中也强制轮换（连续说话的
    /// 兜底）。建议 `soft_ms <= hard_ms/2`。
    Auto { soft_ms: u32, hard_ms: u32 },
    /// 标点触发：累计 `soft_ms` 后，出现下一个定稿标点即轮换——标点是
    /// 模型自己的断句信号，比静音门更及时，且 `Final` 以 `。，？` 等
    /// 收尾，行读起来是完整句。累计 `hard_ms` 即使无标点（音乐/噪声/
    /// 连续说话）也强制轮换。建议 `soft_ms <= hard_ms/2`,无特殊要求下，建议首选Punct。
    Punct { soft_ms: u32, hard_ms: u32 },
    /// 永不轮换：状态随会话时长线性增长，仅适合短会话
    Never,
    /// 自定义判定：返回 `true` 即轮换
    Custom(Arc<dyn Fn(&RotateCtx) -> bool + Send + Sync>),
}

impl RotatePolicy {
    pub fn should_rotate(&self, ctx: &RotateCtx) -> bool {
        match self {
            RotatePolicy::Auto { soft_ms, hard_ms } => {
                ctx.stream_ms >= *hard_ms
                    || (ctx.stream_ms >= *soft_ms && ctx.silence_ms >= SILENCE_GATE_MS)
            }
            RotatePolicy::Punct { hard_ms, .. } => ctx.stream_ms >= *hard_ms || self.punct_hit(ctx),
            RotatePolicy::Never => false,
            RotatePolicy::Custom(f) => f(ctx),
        }
    }

    /// 本次轮换是否由标点触发（而非硬上限兜底）——决定切点是否优先
    /// 取标点位置
    pub(crate) fn punct_hit(&self, ctx: &RotateCtx) -> bool {
        match self {
            RotatePolicy::Punct { soft_ms, .. } => {
                ctx.stream_ms >= *soft_ms && ctx.punct_ms.is_some_and(|p| p >= *soft_ms)
            }
            _ => false,
        }
    }
}

impl Default for RotatePolicy {
    fn default() -> Self {
        RotatePolicy::Punct {
            soft_ms: 10_000,
            hard_ms: 20_000,
        }
    }
}

/// 手动实现：`dyn Fn` 无 `Debug`，而 `Config` 需要整体可 Debug
impl fmt::Debug for RotatePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RotatePolicy::Auto { soft_ms, hard_ms } => {
                write!(f, "Auto {{ soft_ms: {soft_ms}, hard_ms: {hard_ms} }}")
            }
            RotatePolicy::Punct { soft_ms, hard_ms } => {
                write!(f, "Punct {{ soft_ms: {soft_ms}, hard_ms: {hard_ms} }}")
            }
            RotatePolicy::Never => f.write_str("Never"),
            RotatePolicy::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(stream_ms: u32, silence_ms: u32) -> RotateCtx {
        RotateCtx {
            stream_ms,
            silence_ms,
            punct_ms: None,
        }
    }

    fn pctx(stream_ms: u32, punct_ms: Option<u32>) -> RotateCtx {
        RotateCtx {
            stream_ms,
            silence_ms: 0,
            punct_ms,
        }
    }

    #[test]
    fn auto_waits_for_silence_after_soft_limit() {
        let p = RotatePolicy::Auto {
            soft_ms: 3000,
            hard_ms: 60_000,
        };
        // 软上限前：静音再久也不轮换
        assert!(!p.should_rotate(&ctx(2999, 10_000)));
        // 软上限后但静音不足门槛：不轮换
        assert!(!p.should_rotate(&ctx(3000, 499)));
        // 软上限后且静音够久：轮换
        assert!(p.should_rotate(&ctx(3000, 500)));
    }

    #[test]
    fn auto_hard_limit_forces_mid_speech() {
        let p = RotatePolicy::Auto {
            soft_ms: 3000,
            hard_ms: 60_000,
        };
        assert!(!p.should_rotate(&ctx(59_999, 0)));
        assert!(p.should_rotate(&ctx(60_000, 0)));
    }

    #[test]
    fn auto_soft_above_hard_degrades_to_hard_only() {
        let p = RotatePolicy::Auto {
            soft_ms: 90_000,
            hard_ms: 60_000,
        };
        assert!(!p.should_rotate(&ctx(59_999, 10_000)));
        assert!(p.should_rotate(&ctx(60_000, 0)));
    }

    #[test]
    fn never_never_rotates() {
        assert!(!RotatePolicy::Never.should_rotate(&ctx(u32::MAX, u32::MAX)));
    }

    #[test]
    fn custom_predicate_decides() {
        let p = RotatePolicy::Custom(Arc::new(|c| c.silence_ms > 1000));
        assert!(!p.should_rotate(&ctx(999_999, 1000)));
        assert!(p.should_rotate(&ctx(0, 1001)));
    }

    #[test]
    fn default_is_auto_15s_20s() {
        let p = RotatePolicy::default();
        assert!(!p.should_rotate(&ctx(14_999, 60_000)));
        assert!(p.should_rotate(&ctx(15_000, 500)));
        assert!(p.should_rotate(&ctx(20_000, 0)));
    }

    #[test]
    fn punct_triggers_only_after_soft_limit() {
        let p = RotatePolicy::Punct {
            soft_ms: 3000,
            hard_ms: 60_000,
        };
        // 软上限前：有标点也不轮换
        assert!(!p.should_rotate(&pctx(2999, Some(2999))));
        // 软上限后但标点出现在软上限之前：不轮换
        assert!(!p.should_rotate(&pctx(3000, Some(2999))));
        // 软上限后且标点在其后：轮换
        assert!(p.should_rotate(&pctx(3000, Some(3000))));
        // 无标点：等硬上限
        assert!(!p.should_rotate(&pctx(59_999, None)));
        // 硬上限：无标点也强制轮换
        assert!(p.should_rotate(&pctx(60_000, None)));
    }

    #[test]
    fn punct_hit_only_for_punct_trigger_not_hard() {
        let p = RotatePolicy::Punct {
            soft_ms: 3000,
            hard_ms: 60_000,
        };
        assert!(p.punct_hit(&pctx(5000, Some(4000))));
        // 硬上限兜底：不是标点触发
        assert!(!p.punct_hit(&pctx(60_000, None)));
        assert!(!p.punct_hit(&pctx(60_000, Some(1000))));
        // 其他策略永远不是标点触发
        let a = RotatePolicy::Auto {
            soft_ms: 1,
            hard_ms: 2,
        };
        assert!(!a.punct_hit(&pctx(60_000, Some(60_000))));
    }
}

//! GPUs that hung or failed while transcribing (#856), remembered in memory
//! for this process, so Auto keeps that GPU off and transcribes on the CPU
//! (or another GPU). A GPU context reset or device loss comes back from ggml
//! as an error, never as something that could be retried safely on the same
//! device: ggml keeps the lost Vulkan device for the process, and asking a
//! hung driver again can hang the whole desktop's GPU again.
//!
//! Unlike Hex, this standalone build keeps the record in memory only: nothing
//! is written to disk, and a restart starts every GPU fresh. Callers that
//! want cross-restart memory can persist [`Failure`] themselves.

use std::sync::Mutex;

use transcribe_cpp::{Backend, Device, DeviceType};

/// At most this many remembered failures; the oldest go first.
const KEPT: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    /// The device: its description, kind, and hardware id.
    pub device: String,
    /// What its driver was when it failed (see [`driver`]).
    pub driver: String,
    /// The error, for the log.
    pub reason: String,
    /// Seconds since the Unix epoch.
    pub at: u64,
    /// It worked, but slower than the CPU (#484), rather than failing.
    pub slow: bool,
}

/// The failures remembered for this process, read once.
static FAILURES: Mutex<Vec<Failure>> = Mutex::new(Vec::new());

/// What counts as "the same driver" here: the running process. Failures are
/// forgotten at restart; Hex keys them on the installed driver instead.
pub fn driver() -> String {
    format!("process-{}", std::process::id())
}

/// `device` as the failure record names it.
pub fn device_key(device: &Device) -> String {
    format!(
        "{} ({}, {})",
        device.description,
        device.kind,
        device.device_id.as_deref().unwrap_or("-")
    )
}

/// Whether `device` failed within this process.
pub fn failed(device: &Device) -> bool {
    failed_key(&device_key(device))
}

fn failed_key(key: &str) -> bool {
    let driver = driver();
    FAILURES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .any(|failure| failure.device == key && failure.driver == driver)
}

/// Remembers that `device` failed.
pub fn record(device: &Device, reason: &str) {
    remember(&device_key(device), reason, false);
}

/// Remembers that `device` ran the model slower than the CPU (#484).
pub fn record_slow(device: &Device, reason: &str) {
    remember(&device_key(device), reason, true);
}

fn remember(key: &str, reason: &str, slow: bool) {
    let failure = Failure {
        device: key.to_string(),
        driver: driver(),
        reason: reason.chars().take(400).collect(),
        at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs()),
        slow,
    };
    if slow {
        tracing::info!(
            device = failure.device,
            reason = failure.reason,
            "the CPU runs the model faster than this GPU; Auto uses the CPU on it for this session"
        );
    } else {
        tracing::warn!(
            device = failure.device,
            reason = failure.reason,
            "the GPU failed while transcribing; Auto transcribes on the CPU on it for this session"
        );
    }
    let mut failures = FAILURES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    failures.retain(|old| old.device != failure.device || old.driver != failure.driver);
    failures.push(failure);
    let excess = failures.len().saturating_sub(KEPT);
    failures.drain(..excess);
}

/// Forgets `device`'s failure, after it worked when chosen outright.
pub fn forget(device: &Device) {
    let key = device_key(device);
    let driver = driver();
    let mut failures = FAILURES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let before = failures.len();
    failures.retain(|failure| failure.device != key || failure.driver != driver);
    if failures.len() != before {
        tracing::info!(device = key, "the GPU worked again; Auto uses it again");
    }
}

/// The GPUs ggml can run on, best first: discrete before integrated.
pub fn gpus() -> Vec<Device> {
    let mut gpus = transcribe_cpp::devices()
        .into_iter()
        .filter(|device| matches!(device.device_type, DeviceType::Gpu | DeviceType::Igpu))
        .collect::<Vec<_>>();
    gpus.sort_by_key(|device| device.device_type != DeviceType::Gpu);
    gpus
}

/// The backend that runs on `device` alone.
pub fn backend(device: &Device) -> Option<Backend> {
    match device.kind.as_str() {
        "vulkan" => Some(Backend::Vulkan),
        "cuda" => Some(Backend::Cuda),
        "rocm" => Some(Backend::Rocm),
        "metal" => Some(Backend::Metal),
        _ => None,
    }
}

/// Whether `error` says the device failed, rather than the request or the
/// model: the errors a hung or lost GPU, or one out of memory, returns.
pub fn device_error(error: &eyre::Report) -> bool {
    use transcribe_cpp::Error;
    error.chain().any(|cause| {
        cause.downcast_ref::<Error>().is_some_and(|error| {
            matches!(
                error,
                Error::Backend(_) | Error::OutOfMemory(_) | Error::ModelLoad(_) | Error::Other(_)
            )
        })
    })
}

/// Whether `error` says the GPU ran out of memory. That is not a driver
/// fault: another copy of a model can fill a small GPU, so the GPU is not
/// remembered as failed for it (#1165).
pub fn out_of_memory(error: &eyre::Report) -> bool {
    error.chain().any(|cause| {
        if matches!(
            cause.downcast_ref::<transcribe_cpp::Error>(),
            Some(transcribe_cpp::Error::OutOfMemory(_))
        ) {
            return true;
        }
        let message = cause.to_string().to_ascii_lowercase();
        message.contains("out of memory")
            || message.contains("outofdevicememory")
            || message.contains("out_of_device_memory")
            || message.contains("failed to allocate")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_device_errors_count() {
        use eyre::Context;
        let lost: eyre::Result<()> = Err(transcribe_cpp::Error::Backend("device lost".into()))
            .context("transcription failed");
        assert!(device_error(&lost.unwrap_err()));
        let short: eyre::Result<()> =
            Err(transcribe_cpp::Error::InputTooShort("x".into())).context("transcription failed");
        assert!(!device_error(&short.unwrap_err()));
        assert!(!device_error(&eyre::eyre!("cancelled")));
    }

    #[test]
    fn out_of_memory_is_not_a_driver_fault() {
        use eyre::Context;
        let full: eyre::Result<()> =
            Err(transcribe_cpp::Error::OutOfMemory("nope".into())).context("load");
        assert!(out_of_memory(&full.unwrap_err()));
        let other: eyre::Result<()> =
            Err(transcribe_cpp::Error::Backend("device lost".into())).context("load");
        assert!(!out_of_memory(&other.unwrap_err()));
    }

    // `transcribe_cpp::Device` cannot be constructed outside the library (it
    // owns a native handle), so the remember/forget path is tested by key.
    #[test]
    fn a_failed_gpu_is_remembered_and_forgotten() {
        let key = "test gpu (vulkan, 1)";
        assert!(!failed_key(key));
        remember(key, "device lost", false);
        assert!(failed_key(key));
        let mut failures = FAILURES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        failures.retain(|failure| failure.device != key);
        drop(failures);
        assert!(!failed_key(key));
    }

    #[test]
    fn failures_stay_bounded() {
        for i in 0..(KEPT + 4) {
            remember(&format!("bounded gpu {i}"), "boom", false);
        }
        let remembered = FAILURES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        assert_eq!(remembered, KEPT);
    }
}

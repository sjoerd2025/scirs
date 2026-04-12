//! Tracy profiler integration.
//!
//! Enable the `tracy` cargo feature to activate real Tracy profiling via the
//! `tracy-client` crate. Without the feature, all types and functions compile
//! to zero-cost no-ops with no external dependencies.
//!
//! # Usage
//!
//! ```rust,no_run
//! use scirs2_core::profiling::tracy::TracyClient;
//!
//! let client = TracyClient::new();
//! if client.is_active() {
//!     client.message("profiling enabled");
//! }
//! {
//!     let _span = client.span("my_operation");
//!     // work here
//! } // span ends on drop
//! ```

// ---------------------------------------------------------------------------
// Tracy span RAII guard
// ---------------------------------------------------------------------------

/// A Tracy profiling span that ends (is emitted to the profiler) when dropped.
///
/// Obtain one via [`TracyClient::span`].
pub struct TracySpan {
    #[cfg(feature = "tracy")]
    _inner: tracy_client::Span,
    #[cfg(not(feature = "tracy"))]
    _phantom: (),
}

// ---------------------------------------------------------------------------
// TracyClient
// ---------------------------------------------------------------------------

/// Handle to the Tracy profiler client.
///
/// Construct once at application start with [`TracyClient::new`] and keep the
/// handle alive for the duration of profiling.  All methods are safe no-ops
/// when the `tracy` feature is not enabled.
pub struct TracyClient {
    #[cfg(feature = "tracy")]
    inner: tracy_client::Client,
    active: bool,
}

impl TracyClient {
    /// Initialise the Tracy client.
    ///
    /// When the `tracy` feature is enabled this starts the underlying C Tracy
    /// runtime.  When the feature is absent this is a pure no-op constructor.
    pub fn new() -> Self {
        #[cfg(feature = "tracy")]
        {
            TracyClient {
                inner: tracy_client::Client::start(),
                active: true,
            }
        }
        #[cfg(not(feature = "tracy"))]
        {
            TracyClient { active: false }
        }
    }

    /// Returns `true` when Tracy profiling is active (i.e. the `tracy` feature
    /// is enabled and the client was started successfully).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Begin a named profiling zone.  The returned [`TracySpan`] ends the zone
    /// when dropped.
    ///
    /// The `name` string is passed to `span_alloc` and may be a runtime value.
    /// When the `tracy` feature is disabled this is a zero-cost no-op.
    #[inline]
    pub fn span(&self, name: &str) -> TracySpan {
        #[cfg(feature = "tracy")]
        {
            // span_alloc accepts runtime strings; callstack depth 0 = no stack collection.
            let span = self.inner.clone().span_alloc(Some(name), name, "", 0, 0);
            TracySpan { _inner: span }
        }
        #[cfg(not(feature = "tracy"))]
        {
            let _ = name;
            TracySpan { _phantom: () }
        }
    }

    /// Mark a named secondary frame boundary.
    ///
    /// This uses `FrameName::new_leak` to accept runtime `&str` values.
    /// When the `tracy` feature is disabled this is a no-op.
    #[inline]
    pub fn frame_mark(&self, name: &str) {
        #[cfg(feature = "tracy")]
        {
            let frame_name = tracy_client::FrameName::new_leak(name.to_owned());
            self.inner.secondary_frame_mark(frame_name);
        }
        #[cfg(not(feature = "tracy"))]
        let _ = name;
    }

    /// Emit a free-form message to the Tracy log.
    ///
    /// When the `tracy` feature is disabled this is a no-op.
    #[inline]
    pub fn message(&self, msg: &str) {
        #[cfg(feature = "tracy")]
        {
            // callstack depth 0 = no callstack collection
            self.inner.message(msg, 0);
        }
        #[cfg(not(feature = "tracy"))]
        let _ = msg;
    }
}

impl Default for TracyClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Macro convenience
// ---------------------------------------------------------------------------

/// Create a Tracy span in the current scope.
///
/// The span ends when the binding goes out of scope.
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_core::profiling::tracy::TracyClient;
/// use scirs2_core::tracy_span;
///
/// let client = TracyClient::new();
/// tracy_span!(client, "my_operation");
/// // work here — span ends at end of block
/// ```
#[macro_export]
macro_rules! tracy_span {
    ($client:expr, $name:expr) => {
        let _tracy_span_guard = $client.span($name);
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracy_client_default_features() {
        // Must succeed regardless of whether the `tracy` feature is enabled.
        let client = TracyClient::new();

        // Without the `tracy` feature (default), the client should be inactive.
        #[cfg(not(feature = "tracy"))]
        assert!(
            !client.is_active(),
            "TracyClient should be inactive without the tracy feature"
        );

        // With the `tracy` feature active, the client should report active.
        #[cfg(feature = "tracy")]
        assert!(
            client.is_active(),
            "TracyClient should be active with the tracy feature"
        );
    }

    #[test]
    fn test_tracy_span_drop() {
        let client = TracyClient::new();
        {
            let _span = client.span("test_span_drop");
            // span is live here
        }
        // span dropped — no panic
    }

    #[test]
    fn test_tracy_frame_mark() {
        let client = TracyClient::new();
        // Must not panic regardless of feature flag.
        client.frame_mark("test_frame");
    }

    #[test]
    fn test_tracy_message() {
        let client = TracyClient::new();
        // Must not panic regardless of feature flag.
        client.message("test message from tracy integration test");
    }

    #[test]
    fn test_tracy_default_impl() {
        let client = TracyClient::default();
        // Default should produce the same result as new().
        #[cfg(not(feature = "tracy"))]
        assert!(!client.is_active());
    }

    #[test]
    fn test_tracy_span_macro() {
        let client = TracyClient::new();
        tracy_span!(client, "macro_test_span");
        // No panic = success
    }
}

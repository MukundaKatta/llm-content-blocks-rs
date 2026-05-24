//! # llm-content-blocks
//!
//! Typed fluent builder for [Anthropic Messages-API][1] content blocks.
//!
//! The Anthropic Messages API takes a list of content blocks per message:
//! `text`, `image`, `tool_use`, `tool_result`, `document`. The shapes are
//! simple but fiddly, especially when you stitch them together
//! programmatically. [`Blocks`] is a fluent builder that emits the exact
//! JSON shape the API expects, with no SDK dependency.
//!
//! [1]: https://docs.anthropic.com/en/api/messages
//!
//! ## Serialization approach
//!
//! Variants of [`ContentBlock`] derive [`serde::Serialize`] with
//! `#[serde(tag = "type", rename_all = "snake_case")]`. This puts the
//! `"type": "..."` discriminator alongside the variant fields, which is
//! exactly the dict shape Anthropic expects. `cache_control` and
//! `is_error` are skipped when absent / `false` so the produced JSON
//! is byte-equivalent to the Python reference library's output.
//!
//! ## Quick example
//!
//! ```
//! use llm_content_blocks::Blocks;
//!
//! let mut b = Blocks::new();
//! b.text("Look at this:")
//!     .image_b64(b"\x89PNG", "image/png").unwrap()
//!     .text("What is it?");
//! let content = b.build();
//!
//! assert_eq!(content.len(), 3);
//! ```
//!
//! ## Wrap as a full message
//!
//! ```
//! use llm_content_blocks::Blocks;
//!
//! let mut b = Blocks::new();
//! b.text("Hi");
//! let msg = b.build_message("user");
//! assert_eq!(msg.role, "user");
//! assert_eq!(msg.content.len(), 1);
//! ```
//!
//! ## One-shot tool result
//!
//! ```
//! use llm_content_blocks::Blocks;
//! use serde_json::json;
//!
//! let block = Blocks::tool_result_block("toolu_1", json!("the answer"), false);
//! ```

#![deny(missing_docs)]

use std::fmt;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::Serialize;
use serde_json::Value;

/// Image media types accepted by the Anthropic Messages API.
///
/// Slice form keeps the dependency footprint minimal (no `HashSet`
/// allocation for a four-element list).
pub const VALID_IMAGE_MEDIA_TYPES: &[&str] =
    &["image/jpeg", "image/png", "image/gif", "image/webp"];

/// Document media types accepted by the Anthropic Messages API.
pub const VALID_DOCUMENT_MEDIA_TYPES: &[&str] = &["application/pdf", "text/plain"];

/// Cache-control marker for a single content block.
///
/// Only `"ephemeral"` is currently supported on Anthropic's prompt cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
    /// Anthropic prompt-cache marker; serializes as `{"type": "ephemeral"}`.
    Ephemeral,
}

/// Source for an `image` content block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Inline base64-encoded image bytes.
    Base64 {
        /// Image MIME type, must be in [`VALID_IMAGE_MEDIA_TYPES`].
        media_type: String,
        /// Base64-encoded payload.
        data: String,
    },
    /// Remote image fetched by URL.
    Url {
        /// Fully qualified URL.
        url: String,
    },
}

/// Source for a `document` content block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Inline base64-encoded document bytes.
    Base64 {
        /// Document MIME type, must be in [`VALID_DOCUMENT_MEDIA_TYPES`].
        media_type: String,
        /// Base64-encoded payload.
        data: String,
    },
}

/// One entry in the `content` array of an Anthropic message.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text block.
    Text {
        /// Body text.
        text: String,
        /// Optional cache-control marker.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Image block (base64 or URL source).
    Image {
        /// Image source (`base64` or `url`).
        source: ImageSource,
        /// Optional cache-control marker.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Model-issued tool call.
    ToolUse {
        /// Provider-assigned tool-call id (e.g. `toolu_…`).
        id: String,
        /// Tool name as registered with the model.
        name: String,
        /// JSON-shaped tool input.
        input: Value,
    },
    /// User-issued tool result corresponding to a previous tool_use.
    ToolResult {
        /// Id of the tool_use block this is responding to.
        tool_use_id: String,
        /// Result payload (any JSON value: string, list of blocks, etc).
        content: Value,
        /// Whether the result represents an error to the model.
        #[serde(skip_serializing_if = "is_false")]
        is_error: bool,
    },
    /// Document block (PDF, plain text).
    Document {
        /// Document source.
        source: DocumentSource,
        /// Optional cache-control marker.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// A full `{role, content}` message envelope.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Message {
    /// `"user"` or `"assistant"`.
    pub role: String,
    /// The accumulated content blocks.
    pub content: Vec<ContentBlock>,
}

/// Errors returned by builder methods that validate input up front.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockError {
    /// Image media type was not in [`VALID_IMAGE_MEDIA_TYPES`].
    UnsupportedImageMediaType(String),
    /// Document media type was not in [`VALID_DOCUMENT_MEDIA_TYPES`].
    UnsupportedDocumentMediaType(String),
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockError::UnsupportedImageMediaType(t) => write!(
                f,
                "unsupported image media_type {:?}; expected one of {:?}",
                t, VALID_IMAGE_MEDIA_TYPES
            ),
            BlockError::UnsupportedDocumentMediaType(t) => write!(
                f,
                "unsupported document media_type {:?}; expected one of {:?}",
                t, VALID_DOCUMENT_MEDIA_TYPES
            ),
        }
    }
}

impl std::error::Error for BlockError {}

/// Fluent builder for a list of Anthropic content blocks.
///
/// All chained appenders return `&mut Self`; finalize with
/// [`Blocks::build`] (consumes the builder) or [`Blocks::build_message`].
#[derive(Debug, Default, Clone)]
pub struct Blocks {
    inner: Vec<ContentBlock>,
}

impl Blocks {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Append a plain text block.
    pub fn text(&mut self, s: impl Into<String>) -> &mut Self {
        self.inner.push(ContentBlock::Text {
            text: s.into(),
            cache_control: None,
        });
        self
    }

    /// Append a text block carrying a `cache_control` marker.
    pub fn text_with_cache(
        &mut self,
        s: impl Into<String>,
        cache_control: CacheControl,
    ) -> &mut Self {
        self.inner.push(ContentBlock::Text {
            text: s.into(),
            cache_control: Some(cache_control),
        });
        self
    }

    /// Append a base64-encoded inline image block.
    ///
    /// `data` is the raw image bytes; the builder base64-encodes them
    /// internally so callers never have to think about the encoding.
    ///
    /// Returns [`BlockError::UnsupportedImageMediaType`] if `media_type`
    /// is not in [`VALID_IMAGE_MEDIA_TYPES`].
    pub fn image_b64(
        &mut self,
        data: &[u8],
        media_type: impl Into<String>,
    ) -> Result<&mut Self, BlockError> {
        let media_type = media_type.into();
        if !VALID_IMAGE_MEDIA_TYPES.contains(&media_type.as_str()) {
            return Err(BlockError::UnsupportedImageMediaType(media_type));
        }
        let encoded = BASE64_STANDARD.encode(data);
        self.inner.push(ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type,
                data: encoded,
            },
            cache_control: None,
        });
        Ok(self)
    }

    /// Same as [`Blocks::image_b64`] but also tags the block with a
    /// `cache_control` marker.
    pub fn image_b64_with_cache(
        &mut self,
        data: &[u8],
        media_type: impl Into<String>,
        cache_control: CacheControl,
    ) -> Result<&mut Self, BlockError> {
        let media_type = media_type.into();
        if !VALID_IMAGE_MEDIA_TYPES.contains(&media_type.as_str()) {
            return Err(BlockError::UnsupportedImageMediaType(media_type));
        }
        let encoded = BASE64_STANDARD.encode(data);
        self.inner.push(ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type,
                data: encoded,
            },
            cache_control: Some(cache_control),
        });
        Ok(self)
    }

    /// Append an image block that references a remote URL.
    pub fn image_url(&mut self, url: impl Into<String>) -> &mut Self {
        self.inner.push(ContentBlock::Image {
            source: ImageSource::Url { url: url.into() },
            cache_control: None,
        });
        self
    }

    /// Append a `tool_use` block. `input` is taken as a `serde_json::Value`
    /// so callers may use the [`serde_json::json!`] macro or pass any
    /// type implementing `Into<Value>`.
    pub fn tool_use(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        input: Value,
    ) -> &mut Self {
        self.inner.push(ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        });
        self
    }

    /// Append a `tool_result` block.
    pub fn tool_result(
        &mut self,
        tool_use_id: impl Into<String>,
        content: Value,
        is_error: bool,
    ) -> &mut Self {
        self.inner.push(ContentBlock::ToolResult {
            tool_use_id: tool_use_id.into(),
            content,
            is_error,
        });
        self
    }

    /// Append a base64-encoded `document` block. Defaults to PDF when
    /// no media type is supplied (see [`Blocks::document_pdf_b64`]).
    ///
    /// Returns [`BlockError::UnsupportedDocumentMediaType`] if
    /// `media_type` is not in [`VALID_DOCUMENT_MEDIA_TYPES`].
    pub fn document_b64(
        &mut self,
        data: &[u8],
        media_type: impl Into<String>,
    ) -> Result<&mut Self, BlockError> {
        let media_type = media_type.into();
        if !VALID_DOCUMENT_MEDIA_TYPES.contains(&media_type.as_str()) {
            return Err(BlockError::UnsupportedDocumentMediaType(media_type));
        }
        let encoded = BASE64_STANDARD.encode(data);
        self.inner.push(ContentBlock::Document {
            source: DocumentSource::Base64 {
                media_type,
                data: encoded,
            },
            cache_control: None,
        });
        Ok(self)
    }

    /// Convenience wrapper for the common `application/pdf` case.
    pub fn document_pdf_b64(&mut self, data: &[u8]) -> &mut Self {
        // safe: "application/pdf" is in the allowlist
        let _ = self.document_b64(data, "application/pdf");
        self
    }

    /// Splice an existing iterable of content blocks into the builder.
    pub fn extend<I>(&mut self, blocks: I) -> &mut Self
    where
        I: IntoIterator<Item = ContentBlock>,
    {
        self.inner.extend(blocks);
        self
    }

    /// Drain the builder and return the accumulated block list.
    ///
    /// Takes `&mut self` (rather than `self`) so it composes naturally
    /// with the `&mut Self`-returning fluent chain (e.g.
    /// `Blocks::new().text("hi").build()`). The builder is left empty
    /// after the call; calling `.build()` twice yields `[blocks, []]`.
    pub fn build(&mut self) -> Vec<ContentBlock> {
        std::mem::take(&mut self.inner)
    }

    /// Drain the builder and wrap the accumulated blocks in a
    /// `{role, content}` envelope.
    pub fn build_message(&mut self, role: impl Into<String>) -> Message {
        Message {
            role: role.into(),
            content: std::mem::take(&mut self.inner),
        }
    }

    /// Return the current number of accumulated blocks.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` when no blocks have been appended yet.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    // ---- static one-shot helpers ------------------------------------

    /// Build a single `tool_result` block without going through the
    /// builder. Mirrors the Python `Blocks.tool_result` classmethod.
    pub fn tool_result_block(
        tool_use_id: impl Into<String>,
        content: Value,
        is_error: bool,
    ) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.into(),
            content,
            is_error,
        }
    }
}

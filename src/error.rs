use std::path::{Path, PathBuf};

/// Every fallible path in the crate. Each variant is a real boundary:
/// file I/O, JSON on disk or on the wire, SQLite, an external command, HTTP, user input.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{}: {source}", path.display())]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("bad JSON from {what}: {source}")]
    Decode {
        what: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("index: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("`{cmd}` failed: {detail}")]
    Command { cmd: String, detail: String },
    #[error("HTTP {status} from {url}")]
    Http { url: String, status: u16 },
    #[error("not a paper reference: {0:?} (want an arXiv id or URL, or a lowercase slug)")]
    BadRef(String),
    #[error("not a tag: {0:?} (lowercase letters, digits, '-')")]
    BadTag(String),
    #[error("no paper {0} in the library")]
    NotFound(String),
    #[error("{0} is on neither arXiv nor the Hugging Face Hub")]
    Unknown(String),
    #[error("no text for {id}: {why}")]
    NoText { id: String, why: String },
    #[error("{0} is already in the library")]
    Exists(String),
    #[error("no library at {} (run `paperdb init`, or set PAPERDB_LIBRARY)", .0.display())]
    NoLibrary(PathBuf),
    #[error("{0}")]
    Usage(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Attach the path to an I/O error, at the call site where it is known.
pub(crate) fn io<T>(r: std::io::Result<T>, path: &Path) -> Result<T> {
    r.map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}

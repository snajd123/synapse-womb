//! Market data source abstraction.
//!
//! Provides a unified `Source` enum that dispatches to either WebSocket
//! (default, free tier) or gRPC (Yellowstone, paid tier) backends.

#[cfg(feature = "websocket")]
pub mod websocket;

#[cfg(feature = "grpc")]
pub mod grpc;

#[cfg(feature = "websocket")]
pub use websocket::WebSocketSource;

#[cfg(feature = "grpc")]
pub use grpc::GrpcSource;

use anyhow::Result;
use tokio::sync::mpsc;

/// A single account data update from the network.
pub struct AccountUpdate {
    /// Account public key (base58).
    pub pubkey: String,
    /// Slot number.
    pub slot: u64,
    /// Raw account data bytes.
    pub data: Vec<u8>,
}

/// Configuration for connecting to a market data source.
pub struct MarketConfig {
    /// Endpoint URL (wss:// for WebSocket, https:// for gRPC).
    pub endpoint: String,
    /// Optional API key (x-token for gRPC; baked into URL for WebSocket).
    pub api_key: Option<String>,
    /// Account addresses to subscribe to (base58).
    pub accounts: Vec<String>,
}

/// Data source backend.
#[derive(Clone)]
pub enum Source {
    #[cfg(feature = "websocket")]
    WebSocket(WebSocketSource),
    #[cfg(feature = "grpc")]
    Grpc(GrpcSource),
}

impl Source {
    /// Connect and stream account updates into the provided channel.
    /// Returns when the connection drops or an error occurs.
    pub async fn stream(
        &self,
        config: &MarketConfig,
        tx: mpsc::Sender<AccountUpdate>,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "websocket")]
            Source::WebSocket(ws) => ws.stream(config, tx).await,
            #[cfg(feature = "grpc")]
            Source::Grpc(g) => g.stream(config, tx).await,
        }
    }
}

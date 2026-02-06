//! Binary record format for market data capture.
//!
//! Each record: [slot: u64 LE][amm_data: 1024 bytes][coin_amount: u64 LE][pc_amount: u64 LE]
//! Total: 1048 bytes per record.

use std::io::{self, Read, Write};

pub const AMM_DATA_SIZE: usize = 1024;
pub const RECORD_SIZE: usize = 8 + AMM_DATA_SIZE + 8 + 8; // 1048

#[derive(Clone)]
pub struct Record {
    pub slot: u64,
    pub amm_data: [u8; AMM_DATA_SIZE],
    pub coin_amount: u64,
    pub pc_amount: u64,
}

impl Record {
    pub fn from_raw(slot: u64, raw: &[u8], coin_amount: u64, pc_amount: u64) -> Self {
        let mut amm_data = [0u8; AMM_DATA_SIZE];
        let len = raw.len().min(AMM_DATA_SIZE);
        amm_data[..len].copy_from_slice(&raw[..len]);
        Self { slot, amm_data, coin_amount, pc_amount }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(RECORD_SIZE);
        buf.extend_from_slice(&self.slot.to_le_bytes());
        buf.extend_from_slice(&self.amm_data);
        buf.extend_from_slice(&self.coin_amount.to_le_bytes());
        buf.extend_from_slice(&self.pc_amount.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < RECORD_SIZE {
            return Err("buffer too small for record");
        }
        let slot = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let mut amm_data = [0u8; AMM_DATA_SIZE];
        amm_data.copy_from_slice(&bytes[8..8 + AMM_DATA_SIZE]);
        let co = 8 + AMM_DATA_SIZE;
        let coin_amount = u64::from_le_bytes(bytes[co..co + 8].try_into().unwrap());
        let pc_amount = u64::from_le_bytes(bytes[co + 8..co + 16].try_into().unwrap());
        Ok(Self { slot, amm_data, coin_amount, pc_amount })
    }

    pub fn price(&self, coin_decimals: u8, pc_decimals: u8) -> f64 {
        if self.coin_amount == 0 { return 0.0; }
        let coin = self.coin_amount as f64 / 10f64.powi(coin_decimals as i32);
        let pc = self.pc_amount as f64 / 10f64.powi(pc_decimals as i32);
        pc / coin
    }
}

pub fn write_record<W: Write>(w: &mut W, record: &Record) -> io::Result<()> {
    w.write_all(&record.to_bytes())
}

pub fn read_record<R: Read>(r: &mut R) -> io::Result<Option<Record>> {
    let mut buf = vec![0u8; RECORD_SIZE];
    match r.read_exact(&mut buf) {
        Ok(()) => Ok(Some(Record::from_bytes(&buf).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, e)
        })?)),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

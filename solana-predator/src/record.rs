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

// --- DualRecord: Raydium + Orca in a single record ---

pub const DUAL_RECORD_SIZE: usize = 8 + AMM_DATA_SIZE + 8 + 8 + AMM_DATA_SIZE + 8 + 8; // 2088

#[derive(Clone)]
pub struct DualRecord {
    pub slot: u64,
    pub ray_amm_data: [u8; AMM_DATA_SIZE],
    pub ray_coin: u64,
    pub ray_pc: u64,
    pub orca_data: [u8; AMM_DATA_SIZE],
    pub orca_coin: u64,
    pub orca_pc: u64,
}

impl DualRecord {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(DUAL_RECORD_SIZE);
        buf.extend_from_slice(&self.slot.to_le_bytes());
        buf.extend_from_slice(&self.ray_amm_data);
        buf.extend_from_slice(&self.ray_coin.to_le_bytes());
        buf.extend_from_slice(&self.ray_pc.to_le_bytes());
        buf.extend_from_slice(&self.orca_data);
        buf.extend_from_slice(&self.orca_coin.to_le_bytes());
        buf.extend_from_slice(&self.orca_pc.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < DUAL_RECORD_SIZE {
            return Err("buffer too small for dual record");
        }
        let slot = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let mut ray_amm_data = [0u8; AMM_DATA_SIZE];
        ray_amm_data.copy_from_slice(&bytes[8..8 + AMM_DATA_SIZE]);
        let o1 = 8 + AMM_DATA_SIZE;
        let ray_coin = u64::from_le_bytes(bytes[o1..o1 + 8].try_into().unwrap());
        let ray_pc = u64::from_le_bytes(bytes[o1 + 8..o1 + 16].try_into().unwrap());
        let o2 = o1 + 16;
        let mut orca_data = [0u8; AMM_DATA_SIZE];
        orca_data.copy_from_slice(&bytes[o2..o2 + AMM_DATA_SIZE]);
        let o3 = o2 + AMM_DATA_SIZE;
        let orca_coin = u64::from_le_bytes(bytes[o3..o3 + 8].try_into().unwrap());
        let orca_pc = u64::from_le_bytes(bytes[o3 + 8..o3 + 16].try_into().unwrap());
        Ok(Self { slot, ray_amm_data, ray_coin, ray_pc, orca_data, orca_coin, orca_pc })
    }

    pub fn ray_record(&self) -> Record {
        Record { slot: self.slot, amm_data: self.ray_amm_data, coin_amount: self.ray_coin, pc_amount: self.ray_pc }
    }

    pub fn orca_record(&self) -> Record {
        Record { slot: self.slot, amm_data: self.orca_data, coin_amount: self.orca_coin, pc_amount: self.orca_pc }
    }

    pub fn ray_price(&self, coin_dec: u8, pc_dec: u8) -> f64 {
        self.ray_record().price(coin_dec, pc_dec)
    }

    pub fn orca_price(&self, coin_dec: u8, pc_dec: u8) -> f64 {
        self.orca_record().price(coin_dec, pc_dec)
    }
}

pub fn write_dual_record<W: Write>(w: &mut W, record: &DualRecord) -> io::Result<()> {
    w.write_all(&record.to_bytes())
}

pub fn read_dual_record<R: Read>(r: &mut R) -> io::Result<Option<DualRecord>> {
    let mut buf = vec![0u8; DUAL_RECORD_SIZE];
    match r.read_exact(&mut buf) {
        Ok(()) => Ok(Some(DualRecord::from_bytes(&buf).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, e)
        })?)),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
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

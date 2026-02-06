use solana_predator::record::{Record, RECORD_SIZE, AMM_DATA_SIZE, write_record, read_record};

#[test]
fn test_record_size_is_1048() {
    assert_eq!(RECORD_SIZE, 1048);
}

#[test]
fn test_amm_data_size_is_1024() {
    assert_eq!(AMM_DATA_SIZE, 1024);
}

#[test]
fn test_record_to_bytes_roundtrip() {
    let record = Record {
        slot: 123456789,
        amm_data: {
            let mut d = [0u8; AMM_DATA_SIZE];
            d[0] = 0xAB;
            d[1023] = 0xCD;
            d
        },
        coin_amount: 50_000_000_000_000,
        pc_amount: 7_000_000_000_000,
    };
    let bytes = record.to_bytes();
    assert_eq!(bytes.len(), RECORD_SIZE);
    let decoded = Record::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.slot, 123456789);
    assert_eq!(decoded.amm_data[0], 0xAB);
    assert_eq!(decoded.amm_data[1023], 0xCD);
    assert_eq!(decoded.coin_amount, 50_000_000_000_000);
    assert_eq!(decoded.pc_amount, 7_000_000_000_000);
}

#[test]
fn test_record_from_bytes_wrong_size() {
    let bytes = vec![0u8; 100];
    assert!(Record::from_bytes(&bytes).is_err());
}

#[test]
fn test_record_zero_padding() {
    let raw_data = vec![1u8, 2, 3, 4, 5];
    let record = Record::from_raw(42, &raw_data, 1000, 2000);
    assert_eq!(record.amm_data[0], 1);
    assert_eq!(record.amm_data[4], 5);
    assert_eq!(record.amm_data[5], 0);
    assert_eq!(record.amm_data[1023], 0);
    assert_eq!(record.slot, 42);
    assert_eq!(record.coin_amount, 1000);
    assert_eq!(record.pc_amount, 2000);
}

#[test]
fn test_record_truncation() {
    let raw_data = vec![0xFF; 2000];
    let record = Record::from_raw(99, &raw_data, 500, 600);
    assert_eq!(record.amm_data[0], 0xFF);
    assert_eq!(record.amm_data[1023], 0xFF);
    assert_eq!(record.slot, 99);
}

#[test]
fn test_price_from_record() {
    let record = Record {
        slot: 1,
        amm_data: [0u8; AMM_DATA_SIZE],
        coin_amount: 1_000_000_000,
        pc_amount: 150_000_000,
    };
    let price = record.price(9, 6);
    assert!((price - 150.0).abs() < 0.001);
}

#[test]
fn test_write_and_read_multiple_records() {
    use std::io::Cursor;
    let mut buf = Vec::new();
    let r1 = Record::from_raw(100, &[1, 2, 3], 1000, 2000);
    let r2 = Record::from_raw(200, &[4, 5, 6], 3000, 4000);
    write_record(&mut buf, &r1).unwrap();
    write_record(&mut buf, &r2).unwrap();
    let mut cursor = Cursor::new(&buf);
    let d1 = read_record(&mut cursor).unwrap().unwrap();
    let d2 = read_record(&mut cursor).unwrap().unwrap();
    let d3 = read_record(&mut cursor).unwrap();
    assert_eq!(d1.slot, 100);
    assert_eq!(d2.slot, 200);
    assert!(d3.is_none());
}

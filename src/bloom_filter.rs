use std::{cmp::Ordering, collections::HashMap};

const BLOOM_WIDTH: usize = 1 << 27;
const BITS_PER_ENTRY: usize = 12;

#[derive(Debug, Clone, Default)]
pub(crate) struct KmerFilter {
    buf_size: u64,
    buffer: Vec<u64>,
    counts: HashMap<u64, u16>,
    min_count: u16,
}

impl KmerFilter {
    #[inline(always)]
    fn reduce(key: u64, range: u64) -> u64 {
        (((key as u128) * (range as u128)) >> 64) as u64
    }

    #[inline(always)]
    fn cheap_mix(key: u64) -> u64 {
        (key ^ (key >> 31)).wrapping_mul(0x85D0_59AA_3331_21CF)
    }

    #[inline(always)]
    fn fingerprint(key: u64) -> u64 {
        1 << (key & 63)
            | 1 << ((key >> 6) & 63)
            | 1 << ((key >> 12) & 63)
            | 1 << ((key >> 18) & 63)
            | 1 << ((key >> 24) & 63)
    }

    #[inline(always)]
    fn location(key: u64, range: u64) -> usize {
        Self::reduce(Self::cheap_mix(key), range) as usize
    }

    fn bloom_add_and_check(&mut self, key: u64) -> bool {
        let fingerprint = Self::fingerprint(key);
        let idx = Self::location(key, self.buf_size);
        let val = &mut self.buffer[idx];
        if *val & fingerprint == fingerprint {
            true
        } else {
            *val |= fingerprint;
            false
        }
    }

    pub(crate) fn new(min_count: usize) -> Self {
        let buf_size =
            f64::round(BLOOM_WIDTH as f64 * (BITS_PER_ENTRY as f64 / 8.0) / (u64::BITS as f64))
                as u64;
        Self {
            buf_size,
            buffer: Vec::new(),
            counts: HashMap::new(),
            min_count: min_count.min(u16::MAX as usize) as u16,
        }
    }

    pub(crate) fn init(&mut self) {
        if self.buffer.is_empty() {
            self.buffer.resize(self.buf_size as usize, 0);
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn clear(&mut self) {
        self.buffer.clear();
        self.counts.clear();
        self.init();
    }

    pub(crate) fn filter(&mut self, hash: u64) -> Ordering {
        match self.min_count {
            0 | 1 => Ordering::Equal,
            2 => {
                if self.bloom_add_and_check(hash) {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            }
            _ => {
                if self.bloom_add_and_check(hash) {
                    let mut count: u16 = 2;
                    self.counts
                        .entry(hash)
                        .and_modify(|curr_cnt| {
                            count = curr_cnt.saturating_add(1);
                            *curr_cnt = count;
                        })
                        .or_insert(count);
                    self.min_count.cmp(&count)
                } else {
                    Ordering::Less
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::KmerFilter;
    use std::cmp::Ordering;

    #[test]
    fn accepts_all_when_count_is_one_or_zero() {
        for count in [0, 1] {
            let mut filter = KmerFilter::new(count);
            filter.init();
            for hash in [11, 11, 17, 23, 17] {
                assert_eq!(filter.filter(hash), Ordering::Equal);
            }
        }
    }

    #[test]
    fn admits_on_second_observation_when_count_is_two() {
        let mut filter = KmerFilter::new(2);
        filter.init();
        assert_eq!(filter.filter(42), Ordering::Less);
        assert_eq!(filter.filter(42), Ordering::Equal);
        assert_eq!(filter.filter(42), Ordering::Equal);
    }

    #[test]
    fn admits_on_threshold_for_larger_counts() {
        let mut filter = KmerFilter::new(4);
        filter.init();
        assert_eq!(filter.filter(9), Ordering::Less);
        assert_eq!(filter.filter(9), Ordering::Greater);
        assert_eq!(filter.filter(9), Ordering::Greater);
        assert_eq!(filter.filter(9), Ordering::Equal);
        assert_eq!(filter.filter(9), Ordering::Less);
    }

    #[test]
    fn clear_resets_state() {
        let mut filter = KmerFilter::new(3);
        filter.init();
        assert_eq!(filter.filter(5), Ordering::Less);
        assert_eq!(filter.filter(5), Ordering::Greater);
        assert_eq!(filter.filter(5), Ordering::Equal);
        filter.clear();
        assert_eq!(filter.filter(5), Ordering::Less);
        assert_eq!(filter.filter(5), Ordering::Greater);
        assert_eq!(filter.filter(5), Ordering::Equal);
    }
}

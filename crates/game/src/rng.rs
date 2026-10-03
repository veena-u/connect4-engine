/// A random number generator, to avoid deps
pub struct Rng(u64); 

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn weighted<T: Copy>(&mut self, items: &[(T, f64)]) -> T {
        let total: f64 = items.iter().map(|&(_, w)| w).sum();
        let mut u = self.unit() * total;
        for &(item, w) in items {
            u -= w;
            if u < 0.0 {
                return item;
            }
        }
        items.last().expect("no items to choose from").0
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.unit(); 
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn gamma(&mut self, shape: f64) -> f64 {
        assert!(shape > 0.0);
        if shape < 1.0 {
            let u = 1.0 - self.unit();
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = 1.0 - self.unit();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    pub fn dirichlet(&mut self, alpha: f64, out: &mut [f32]) {
        for x in out.iter_mut() {
            *x = self.gamma(alpha) as f32;
        }
        let total: f32 = out.iter().sum();
        if total > 0.0 {
            out.iter_mut().for_each(|x| *x /= total);
        } else {
            out.fill(1.0 / out.len() as f32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_has_the_right_mean() {
        let mut rng = Rng::new(1);
        for shape in [0.3, 1.0, 2.5] {
            let n = 50_000;
            let mean = (0..n).map(|_| rng.gamma(shape)).sum::<f64>() / n as f64;
            assert!((mean - shape).abs() < 0.05 * shape.max(1.0), "shape {shape}: mean {mean}");
        }
    }

    #[test]
    fn dirichlet_sums_to_one_with_equal_means() {
        let mut rng = Rng::new(2);
        let mut sample = [0.0f32; 4];
        let mut sums = [0.0f64; 4];
        for _ in 0..20_000 {
            rng.dirichlet(0.5, &mut sample);
            assert!((sample.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(sample.iter().all(|&x| x >= 0.0));
            sums.iter_mut().zip(sample).for_each(|(s, x)| *s += x as f64);
        }
        for s in sums {
            assert!((s / 20_000.0 - 0.25).abs() < 0.01);
        }
    }
}

use core::fmt;
use std::time::Duration;
use std::{
    cmp::{max, min},
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use itertools::izip;

/// Read in x- and y-values from a plaintext data file.
pub fn load_txt(datafile: &PathBuf) -> Result<(Vec<f64>, Vec<f64>), String> {
    let file = match File::open(datafile) {
        Ok(v) => v,
        Err(e) => return Err(format!("Got error when opening {:?}: {}", datafile, e)),
    };
    let reader = BufReader::new(file);

    let (mut x_ray, mut y_ray) = (Vec::new(), Vec::new());
    for line in reader.lines().map_while(Result::ok) {
        let vals: Vec<&str> = line.split(' ').collect();
        if let [x, y] = vals[..] {
            if let (Ok(x), Ok(y)) = (x.parse::<f64>(), y.parse::<f64>()) {
                x_ray.push(x);
                y_ray.push(y);
            } else {
                return Err(format!("Found non-float values in data list: {}, {}", x, y));
            }
        } else {
            return Err(format!(
                "Got malformed data: {}. Data rows must \
                only contain two space-separated values",
                line,
            ));
        }
    }

    Ok((x_ray, y_ray))
}

/// Format a number so only a given number of significant digits are shown.
/// If the number is very large/small, scientific notation will be used.
pub fn g_format(number: f64, sigdig: usize) -> String {
    /// The maximum number of digits before scientific notation is used
    const SCI_THRESHOLD: i64 = 3;

    let digits = number.abs().log10().ceil() as i64;
    if digits > min(sigdig as i64, SCI_THRESHOLD + 1) || digits < -SCI_THRESHOLD {
        format!("{:.num$e}", number, num = sigdig - 1)
    } else {
        format!("{:.num$}", number, num = (sigdig as i64 - digits) as usize)
    }
}

pub fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 1.0 {
        format!("{:.3} ms", seconds * 1000.0)
    } else {
        format!("{:.3} s", seconds)
    }
}

pub fn format_vector(vector: &[f64], sigdig: usize) -> String {
    let mut output_string = "[".to_string();
    for i in 0..vector.len() {
        output_string += &g_format(vector[i], sigdig);
        if i != vector.len() - 1 {
            output_string += ", ";
        }
    }
    output_string + "]"
}

pub fn prettify_list<T: fmt::Display>(list: &[T]) -> String {
    let mut pretty_list = String::new();
    for (i, string) in list.iter().enumerate() {
        pretty_list += &format!("'{}'", string);
        match i {
            i if i < list.len() - 2 => pretty_list += ", ",
            i if i == list.len() - 2 => pretty_list += " and ",
            _ => {}
        }
    }

    pretty_list
}

pub fn format_with_uncertainty(values: &[f64], uncertainties: &[f64]) -> String {
    const UNCERTAINTY_SIGDIG: usize = 1;

    assert_eq!(values.len(), uncertainties.len());

    let mut output_string = "[".to_string();
    for (i, (v, e)) in izip!(values, uncertainties).enumerate() {
        let e_digits = e.abs().log10().ceil() as i64;
        let v_digits = v.abs().log10().ceil() as i64;
        let extra_digits = max(v_digits - e_digits, 0) as usize;

        output_string += &format!(
            "{}±{}",
            g_format(*v, UNCERTAINTY_SIGDIG + extra_digits),
            g_format(*e, UNCERTAINTY_SIGDIG)
        );
        if i != values.len() - 1 {
            output_string += ", ";
        }
    }
    output_string + "]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g_format() {
        assert_eq!(g_format(1426837.0, 4), "1.427e6");
        assert_eq!(g_format(49.279863, 5), "49.280");
        assert_eq!(g_format(2.675289, 3), "2.68");
        assert_eq!(g_format(0.4678, 1), "0.5");
        assert_eq!(g_format(0.0000324, 3), "3.24e-5");
    }

    #[test]
    fn test_format_duration() {
        let duration = Duration::from_secs_f64(13.12736);
        assert_eq!(format_duration(duration), "13.127 s");

        let duration = Duration::from_secs_f64(0.0723847184);
        assert_eq!(format_duration(duration), "72.385 ms");
    }

    #[test]
    fn test_format_vector() {
        let vector = vec![2.13, 9.81, 0.012];
        assert_eq!(format_vector(&vector, 2), "[2.1, 9.8, 0.012]")
    }

    #[test]
    fn test_prettify_list() {
        let list = ["apple", "orange", "banana"];
        assert_eq!(prettify_list(&list), "'apple', 'orange' and 'banana'");

        let list = [1, 15, 30];
        assert_eq!(prettify_list(&list), "'1', '15' and '30'");
    }
}

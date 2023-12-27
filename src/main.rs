use vers::indexes::ivfflat::IVFFlatIndex;
use vers::indexes::lsh::{ANNIndex, Vector};
fn main() {
    println!("Hello, world!");

    let val = Vector([1.0, 2.0, 3.0]);
    println!("{:?}", val);
}

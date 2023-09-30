pub struct HashKey<const N: usize>(pub [u32; N]);
pub struct Vector<const N: usize>(pub [f32; N]);

impl<const N: usize> Vector<N> {
    pub fn subtract_from(&self, other: &Vector<N>) -> Vector<N> {
        let vals = self.0.iter().zip(&other.0).map(|(a, b)| b - a);
        let result: [f32; N] = vals.collect::<Vec<_>>().try_into().unwrap();
        Vector(result)
    }

    pub fn dot_product(&self, other: &Vector<N>) -> f32 {
        self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum::<f32>()
    }

    pub fn average(&self, other: &Vector<N>) -> Vector<N> {
        let vals = self.0.iter().zip(&other.0).map(|(a, b)| (a + b) / 2.0);
        let results: [f32; N] = vals.collect::<Vec<_>>().try_into().unwrap();
        return Vector(results);
    }

    pub fn to_hashkey(&self) -> HashKey<N> {
        let vals = self.0.iter().map(|a| a.to_bits());
        let results: [u32; N] = vals.collect::<Vec<_>>().try_into().unwrap();
        return HashKey(results);
    }
}

struct Hyperplane<const N: usize> {
    coefficients: Vector<N>,
    constant: f32,
}

impl<const N: usize> Hyperplane<N> {
    /// Checks if the given vector is "above" the hyperplane.
    /// The hyperplane is defined by its normal vector (and its constant)
    /// and when the dot product is taken between the normal vector and
    /// the query vector, a positive value means both vectors are in the same
    /// direction, hence "above" - since a.b = |a||b|cos(0), where 0 is the
    /// angle between vectors a and b. If the dot product is negative,
    /// 0 is > 90deg.
    pub fn point_is_above(&self, point: Vector<N>) -> bool {
        return &self.coefficients.dot_product(&point) + self.constant >= 0.0;
    }
}

enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode>),
}

struct InnerNode<const N: usize> {
    hyperplane: Hyperplane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}

struct LeafNode(Vec<usize>);

pub struct ANNIndex<const N: usize> {
    trees: Vec<Node<N>>,
    values: Vec<Vector<N>>,
    ids: Vec<u32>,
}

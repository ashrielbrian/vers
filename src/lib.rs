use rand::seq::{index, SliceRandom};

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

    pub fn squared_euclidean(&self, other: &Vector<N>) -> f32 {
        return self
            .0
            .iter()
            .zip(other.0)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
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
    pub fn point_is_above(&self, point: &Vector<N>) -> bool {
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
    // vector containing all the trees that make up the index, with each element in the vector
    // indicating the root node of the tree
    trees: Vec<Node<N>>,
    // stores all the vectors within the index in a global contiguous location
    values: Vec<Vector<N>>,
    ids: Vec<u32>,
}

impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane(
        indexes: &Vec<usize>,
        all_vecs: &Vec<Vector<N>>,
    ) -> (Hyperplane<N>, Vec<usize>, Vec<usize>) {
        // sample two random vectors from the indexes, and use these two vectors to form a hyperplane boundary
        let samples: Vec<_> = indexes
            .choose_multiple(&mut rand::thread_rng(), 2)
            .collect();

        let (a, b) = (*samples[0], *samples[1]);

        // construct the hyperplane by getting the plane's coefficients (normal vector), and the
        // its constant (the constant term from a plane's cartesian eqn).
        let coefficients = all_vecs[a].subtract_from(&all_vecs[b]);
        let point_on_plane = all_vecs[a].average(&all_vecs[b]);
        let constant = -coefficients.dot_product(&point_on_plane);

        let hyperplane = Hyperplane {
            coefficients,
            constant,
        };

        // assign each index (which points to a specific vector in the embedding matrix) to
        // either `above` or `below` the hyperplane.
        let mut above: Vec<usize> = Vec::new();
        let mut below: Vec<usize> = Vec::new();

        for id in indexes {
            if hyperplane.point_is_above(&all_vecs[*id]) {
                above.push(*id);
            } else {
                below.push(*id);
            }
        }

        return (hyperplane, above, below);
    }

    fn build_a_tree(max_size: usize, indexes: Vec<usize>, all_vecs: &Vec<Vector<N>>) -> Node<N> {
        if indexes.len() < max_size {
            return Node::Leaf(Box::new(LeafNode(indexes)));
        } else {
            let (hyperplane, above, below) = Self::build_hyperplane(&indexes, all_vecs);

            let left_node = Self::build_a_tree(max_size, above, all_vecs);
            let right_node = Self::build_a_tree(max_size, below, all_vecs);

            return Node::Inner(Box::new(InnerNode {
                hyperplane,
                left_node,
                right_node,
            }));
        }
    }
}

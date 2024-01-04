use criterion::{criterion_group, criterion_main, Criterion};
use vers::indexes::ivfflat::IVFFlatIndex;
use vers::utils;

pub fn criterion_benchmark(c: &mut Criterion) {
    const DIM: usize = 300;
    let (wiki, _, _, _) = utils::load_wiki_vector::<DIM>("wiki-news-300d-1M.vec");

    for num_cluster in [5, 10, 25] {
        c.bench_function(format!("IVFFlat: {} clusters", num_cluster).as_str(), |b| {
            b.iter(|| IVFFlatIndex::build_index(num_cluster, 3, 10, &wiki))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

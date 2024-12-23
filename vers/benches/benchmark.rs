use criterion::{criterion_group, criterion_main, Criterion};
use vers::utils;
use vers::IVFFlatIndex;

pub fn criterion_benchmark(c: &mut Criterion) {
    const DIM: usize = 300;
    let (wiki, _, _, _) = utils::load_wiki_vector::<DIM>("../wiki-news-300d-1M.vec");

    for num_cluster in [1, 10] {
        c.bench_function(format!("IVFFlat: {} clusters", num_cluster).as_str(), |b| {
            b.iter(|| IVFFlatIndex::build_index(num_cluster, 2, 5, &wiki))
        });
    }
}

criterion_group! {
    name=benches;
    config=Criterion::default().sample_size(20);
    targets=criterion_benchmark
}
criterion_main!(benches);

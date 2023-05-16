# Model Disagreement Sampler

The model disagreement sampler identifies experimental conditions $\vec{x}' \in X'$ with respect to
a pairwise distance metric between theorist models, $P_{M_{i}}(\hat{y}, \vec{x}')$:

$$
\underset{\vec{x}'}{\arg\max}~(P_{M_{1}}(\hat{y}, \vec{x}') - P_{M_{2}}(\hat{y}, \vec{x}'))^2
$$

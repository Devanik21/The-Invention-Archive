# When Mutual Information Falls, What Has Actually Changed?

## A source-channel audit for aging-related molecular communication

Mutual information is a useful way to quantify how much knowing one variable reduces uncertainty about another. In cell biology, it can measure how much a transcription factor's activity tells us about the expression of its target gene, or how much a molecular input predicts a downstream response. Because aging changes many molecular distributions at once, information-theoretic measurements are attractive tools for studying the loss of biological regulation.

But a single statistic cannot answer every mechanistic question. A decline in mutual information does not, by itself, prove that a biochemical communication channel has become noisier. The distribution of the inputs being sent can change even if the conditional input-output relation remains exactly the same. This distinction is elementary in information theory and consequential in biological interpretation.

Recent work makes the question especially relevant. Emison and colleagues reported in *Cell Reports Physical Science* in September 2026 that, in their analysis of transcriptional regulation and aging, input-distribution mismatch rather than channel corruption drove the observed decline in mutual information. A 2025 study by Sivakumar and colleagues also applied information-theoretic measurements to skeletal-muscle single-cell RNA sequencing across age groups and reported changes in effective biomolecular communication. These are system-specific empirical results, not a universal law of aging. Their broader methodological lesson is that the measured statistic and the biological mechanism must be kept distinct.

This essay presents a small mathematical audit intended to make that distinction operational.

## 1. Source and channel are different objects

Let X represent an input, such as a discretized transcription-factor state, and Y represent a measured response, such as a target-gene expression category. The source distribution p(x) tells us how often different input states occur. The conditional response law W(y|x) tells us how likely each output is when a particular input occurs. Their combination defines the joint observations:

$$
P(x,y)=p(x)W(y|x).
$$

Mutual information is calculated from this joint distribution. It therefore depends on both p and W. Changing the source can change mutual information without changing the conditional response law. Changing the conditional response law can also change mutual information, and in real datasets both changes may occur together.

This is not a technical nuisance that can be ignored after computing a score. It is an identifiability limit. If a statistic depends on two objects but the experiment reports only that statistic, a change in the statistic cannot uniquely tell us which object changed.

## 2. A simple counterexample

Consider a binary communication channel. The input X is either zero or one, and the output Y is the input with an independent probability ε of being flipped. This is a binary symmetric channel. Let the probability that X equals one be p. The output probability is ε + p(1 − 2ε), and the mutual information in bits is:

$$
I_p(W)=H_2(\epsilon+p(1-2\epsilon))-H_2(\epsilon),
$$

where H₂ is binary entropy.

Keep the channel fixed at ε = 0.10. If the input is equally likely to be zero or one, the mutual information is 1 − H₂(0.10), approximately 0.531 bits per use. Now change only the source so that one is sent with probability 0.05. The channel still flips exactly ten percent of input bits on average. Its conditional response rule has not changed. Yet the mutual information drops to about 0.115 bits per use. The output has become more predictable in an unconditional sense because the source mostly sends zero; that does not mean that the channel has become less reliable conditional on the input.

The point is not that gene regulation is a binary symmetric channel. It is not. The point is that the inference from “mutual information decreased” to “the channel degraded” is invalid without additional information about the source distribution.

## 3. Why aging data are particularly vulnerable to this ambiguity

Aging-related molecular datasets rarely compare identical populations under identical conditions. Cell-type proportions can shift. Transcription-factor activity distributions can change. Gene expression may become sparse. Sequencing depth, dropout rates, batch structure, and normalization choices can differ. Some changes reflect biology; others reflect measurement; often their effects interact.

Suppose a regulatory input becomes concentrated in a narrower range in an older cohort. Even if the output distribution conditional on each input state remains the same, the observed input-output mutual information can decline. Conversely, a conditional response could deteriorate while an altered input distribution partially masks that change. The raw mutual-information contrast is useful as a description of the joint distribution, but it does not alone identify the underlying mechanism.

This does not invalidate information theory for aging research. It defines a more precise question for it to answer.

## 4. Standardize to a common source distribution

A practical audit estimates the conditional response law W_g(y|x) separately for each cohort g. Analysts then choose a reference input distribution q(x) that lies within the common observed support and calculate mutual information as though every cohort had that same input law:

$$
I_q(W_g)=\sum_{x,y}q(x)W_g(y|x)\log_2\frac{W_g(y|x)}{\sum_{x'}q(x')W_g(y|x')}.
$$

The standardized statistic asks a clearer comparative question: if the input states occurred with the same reference frequencies, how much information would each cohort's estimated conditional response law transmit under the model? It removes one source of comparability failure, but it is not a complete causal experiment. The reference q must be declared, its support must be observed, and the conclusion can depend on the chosen reference distribution.

There is a compact exact decomposition. Let Y denote the younger cohort and A the older cohort. The natural contrast compares I under each cohort's own source. Add and subtract standardized values under q:

$$
\Delta_{\mathrm{nat}}=[I_{p_A}(W_A)-I_q(W_A)]+[I_q(W_A)-I_q(W_Y)]+[I_q(W_Y)-I_{p_Y}(W_Y)].
$$

This is a telescoping identity. The middle term compares conditional laws under a common source. The first and last terms describe each cohort's deviation from the reference source under its own estimated channel. These terms are reference-dependent and must not be described as a unique causal decomposition. They are an analysis aid, not a biological law.

## 5. What a responsible empirical audit would report

A paper making claims about age-associated communication should report more than a single mutual-information curve. First, it should provide the natural mutual-information estimate and uncertainty intervals. Second, it should show the estimated conditional response law, with sample counts for each input state. Third, it should report common-source standardized results and sensitivity to reasonable choices of q. Fourth, it should examine cell composition, measurement depth, batch effects, dropout, normalization, and the stability of the estimates under resampling or held-out prediction.

Common-support checks are essential. If an input state is never observed in one cohort, its response distribution cannot be estimated from that cohort without extrapolation. A standardization procedure must not silently invent the missing response. The correct options are to restrict the reference support, collect new measurements, or clearly state that the comparison is not identified from the available data.

Where it is technically and ethically appropriate, controlled perturbations can be more informative than observational cohort comparisons because they deliberately vary the input over a common range. But experimental manipulations still require attention to feedback, hidden variables, nonstationarity, off-target effects, and the difference between an intervention's effect and an observed conditional association.

## 6. Interpretation requires restraint

A persistent difference in standardized mutual information supports the claim that the estimated conditional distributions differ under the selected variables and reference source. It does not, on its own, demonstrate that a particular molecular machine has aged, that a signal has become corrupted, or that restoring the measured statistic would restore tissue function. A biological mechanism requires separate evidence, and a therapeutic proposal requires independent validation of efficacy and safety.

The converse also matters. If natural mutual information falls but the standardized comparison is stable, analysts should not simply conclude that the biology is unchanged. The source distribution itself may be biologically meaningful: a cell receiving fewer or less varied regulatory inputs may indeed operate differently. Standardization clarifies which component of the statistical contrast is changing; it does not automatically label that component as harmful or irrelevant.

## 7. The narrow contribution

The accompanying preprint develops the source-channel distinction formally, gives a reference-distribution decomposition, and includes a deterministic Python script that checks a synthetic binary-channel example. The script uses no biological dataset. Its role is to verify the mathematical point and the implementation, not to simulate organismal aging.

The contribution should be evaluated on those terms: a clear identifiability statement, a reproducible counterexample, and an auditable analysis protocol. It is not presented as a new theory of aging or a claim that biology is merely a communication network.

## Conclusion

Information theory can help distinguish different explanations for age-associated molecular changes, but only when the quantities being measured are interpreted precisely. Mutual information is a property of a joint distribution, not a direct meter of channel quality in isolation. Separating source distributions from conditional response laws makes the analysis more transparent and the scientific claims more falsifiable.

That discipline is useful regardless of whether future experiments find that a given pathway changes primarily through altered inputs, altered responses, or both. The objective is not to force every aging phenotype into one information-theoretic narrative. It is to make each inference match what the data actually identify.

**Author:** Devanik Debnath (Devanik21), Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India.

**Research archive:** https://github.com/Devanik21/The-Invention-Archive

**ORCID:** https://orcid.org/0009-0001-8310-6921

**Selected references:** Emison et al. (2026), https://doi.org/10.1016/j.xcrp.2026.103516; Sivakumar et al. (2025), https://doi.org/10.1093/gerona/glaf195; Rhee, Cheong and Levchenko (2012), https://doi.org/10.1088/1478-3975/9/4/045011.

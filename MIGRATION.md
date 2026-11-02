# Shared-core consumer

This repository uses the canonical semcs_core package maintained in the
SeMIS reliability Python repository. pyproject.toml and requirements.txt
identify a fixed Git commit of that dependency. The dependency is installed
automatically with the package; a sibling checkout is only needed for
coordinated development with an explicit PYTHONPATH.

Install this checkout with Git available for the commit-pinned dependency:

    python -m pip install .

Existing source-tree drivers may alternatively install dependencies with
python -m pip install -r requirements.txt before running from the checkout.
The legacy sampler paths import the canonical classes and contain no
independent sampling-kernel implementation. Existing Main/SeMCS constructors
delegate to the single semcs_core.Engine.

Method configuration and evolution state live in G: thresholds, level
probability, Gaussian parameters, run budgets, sampler options and adapted
lambda/HMC time. Historical configuration attributes are G-backed views.
The former H active-target mask is G["active"].

Each method retains its own advance, seed-selection, chain-plan and completion
rules. The common Engine only orchestrates these policies and records frozen
sampled-level G/data history. Native estimators remain separate. The default
aCS uses unit base scale and carries lambda between levels; historical author
settings must be labeled explicitly rather than silently replacing that default.

Hard SuS and soft SeMIS retain their own thresholds and stopping rules.
The driver keeps distinct native SuS evidence, SeMIS MIS evidence and SeMIS
SIS evidence outputs.

Architecture equivalence and source-paper performance comparisons are
separate validation questions. Consult the evidence supplied with the release;
the common engine alone does not establish reproduction of every paper result.

# PaperQA2 for Clinical Trials

PaperQA2 now natively supports querying clinical trials in addition to any documents supplied by the user. It
uses a new tool, the aptly named `clinical_trials_search` tool. Users don't have to provide any clinical
trials to the tool itself, it uses the `clinicaltrials.gov` API to retrieve them on the fly. As of
January 2025, the tool is not enabled by default, but it's easy to configure. Here's an example
where we query only clinical trials, without using any documents:

```python
from paperqa import Settings, agent_query

answer_response = await agent_query(
    query="What drugs have been found to effectively treat Ulcerative Colitis?",
    settings=Settings.from_name("search_only_clinical_trials"),
)

print(answer_response.session.answer)
```

### Output

    Several drugs have been found to effectively treat Ulcerative Colitis (UC),
    targeting different mechanisms of the disease.

    Golimumab, a tumor necrosis factor (TNF) inhibitor marketed as Simponi®, has demonstrated efficacy
    in treating moderate-to-severe UC. Administered subcutaneously, it was shown to maintain clinical
    response through Week 54 in patients, as assessed by the Partial Mayo Score (NCT02092285).

    Mesalazine, an anti-inflammatory drug, is commonly used for UC treatment. In a study comparing
    mesalazine enemas to faecal microbiota transplantation (FMT) for left-sided UC,
    mesalazine enemas (4g daily) were effective in inducing clinical remission (Mayo score ≤ 2) (NCT03104036).

    Antibiotics have also shown potential in UC management. A combination of doxycycline,
    amoxicillin, and metronidazole induced remission in 60-70% of patients with moderate-to-severe
    UC in prior studies. These antibiotics are thought to alter gut microbiota, reducing pathobionts
     and promoting beneficial bacteria (NCT02217722, NCT03986996).

    Roflumilast, a phosphodiesterase-4 (PDE4) inhibitor, is being investigated for mild-to-moderate UC.
    Preliminary findings suggest it may improve disease severity and biochemical markers when
    added to conventional treatments (NCT05684484).

    These treatments highlight diverse therapeutic approaches, including immunosuppression,
    microbiota modulation, and anti-inflammatory mechanisms.

You can see the in-line citations for each clinical trial used as a response for each query. If you'd like
to see more data on the specific contexts that were used to answer the query:

```python
print(answer_response.session.contexts)
```

    [Context(context='The excerpt mentions that a search on ClinicalTrials.gov for clinical trials related to drugs
    treating Ulcerative Colitis yielded 689 trials. However, it does not provide specific information about which
    drugs have been found effective for treating Ulcerative Colitis.', text=Text(text='', name=...

Using `Settings.from_name('search_only_clinical_trials')` is a shortcut, but note that you can easily
add `clinical_trial_search` into any custom `Settings` by just explicitly naming it as a tool:

```python
from pathlib import Path
from paperqa import Settings, agent_query, AgentSetting
from paperqa.agents.tools import DEFAULT_TOOL_NAMES

# you can start with the default list of PaperQA tools
print(DEFAULT_TOOL_NAMES)
# >>> ['paper_search', 'gather_evidence', 'gen_answer', 'reset', 'complete'],

# we can start with a directory with a potentially useful paper in it
print(list(Path("my_papers").iterdir()))

# now let's query using standard tools + clinical_trials
answer_response = await agent_query(
    query="What drugs have been found to effectively treat Ulcerative Colitis?",
    settings=Settings(
        paper_directory="my_papers",
        agent={"tool_names": DEFAULT_TOOL_NAMES + ["clinical_trials_search"]},
    ),
)

# let's check out the formatted answer (with references included)
print(answer_response.session.formatted_answer)
```

    Question: What drugs have been found to effectively treat Ulcerative Colitis?

    Several drugs have been found effective in treating Ulcerative Colitis (UC), with treatment
    strategies varying based on disease severity and extent. For mild-to-moderate UC, 5-aminosalicylic
     acid (5-ASA) is the first-line therapy. Topical 5-ASA, such as mesalazine suppositories (1 g/day),
     is effective for proctitis or distal colitis, inducing remission in 31-80% of patients. Oral mesalazine
     at higher doses (e.g., 4.8 g/day) can accelerate clinical improvement in more extensive disease
     (meier2011currenttreatmentof pages 1-2; meier2011currenttreatmentof pages 3-4).

    For moderate-to-severe cases, corticosteroids are commonly used. Oral steroids like prednisolone
    (40-60 mg/day) or intravenous steroids such as methylprednisolone (60 mg/day) and hydrocortisone
    (400 mg/day) are standard for inducing remission (meier2011currenttreatmentof pages 3-4). Tumor
    necrosis factor (TNF)-α blockers, such as infliximab, are effective for steroid-refractory cases
    (meier2011currenttreatmentof pages 2-3; meier2011currenttreatmentof pages 3-4).

    Immunosuppressive agents, including azathioprine and 6-mercaptopurine, are used for maintenance
    therapy in steroid-dependent or refractory cases (meier2011currenttreatmentof pages 2-3;
    meier2011currenttreatmentof pages 3-4). Antibiotics, such as combinations of penicillin,
    tetracycline, and metronidazole, have shown promise in altering the microbiota and inducing
    remission in some patients, though their efficacy varies (NCT02217722).

    References

    1. (meier2011currenttreatmentof pages 2-3): Johannes Meier and Andreas Sturm. Current treatment
    of ulcerative colitis. World journal of gastroenterology, 17 27:3204-12, 2011.
    URL: https://doi.org/10.3748/wjg.v17.i27.3204, doi:10.3748/wjg.v17.i27.3204.

    2. (meier2011currenttreatmentof pages 3-4): Johannes Meier and Andreas Sturm. Current treatment
    of ulcerative colitis. World journal of gastroenterology, 17 27:3204-12, 2011. URL:
    https://doi.org/10.3748/wjg.v17.i27.3204, doi:10.3748/wjg.v17.i27.3204.

    3. (NCT02217722): Prof. Arie Levine. Use of the Ulcerative Colitis Diet for Induction of
    Remission. Prof. Arie Levine. 2014. ClinicalTrials.gov Identifier: NCT02217722

    4. (meier2011currenttreatmentof pages 1-2): Johannes Meier and Andreas Sturm. Current
    treatment of ulcerative colitis. World journal of gastroenterology, 17 27:3204-12, 2011.
     URL: https://doi.org/10.3748/wjg.v17.i27.3204, doi:10.3748/wjg.v17.i27.3204.

We now see both papers and clinical trials cited in our response. For convenience, we have a
`Settings.from_name` that works as well:

```python
from paperqa import Settings, agent_query

answer_response = await agent_query(
    query="What drugs have been found to effectively treat Ulcerative Colitis?",
    settings=Settings.from_name("clinical_trials"),
)
```

And, this works with the `pqa` cli as well:

```bash
>>> pqa --settings 'search_only_clinical_trials' ask 'what is Ibuprofen effective at treating?'
```

    ...
    [13:29:50] Completing 'what is Ibuprofen effective at treating?' as 'certain'.
            Answer: Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) effective
            in treating various conditions, including pain, inflammation, and fever.
            It is widely used for tension-type
            headaches, with studies showing that ibuprofen sodium provides significant
            pain relief and reduces pain intensity compared to standard ibuprofen and placebo
            over a 3-hour period (NCT01362491).
            Intravenous ibuprofen is effective in managing postoperative pain, particularly
            in orthopedic surgeries, and helps control the inflammatory process. When combined
            with opioids, it reduces opioid
            consumption and associated side effects, making it a key component of
            multimodal analgesia (NCT05401916, NCT01773005).

            Ibuprofen is also effective in pediatric populations as a first-line
            anti-inflammatory and antipyretic agent due to its relatively
            low adverse effects compared to other NSAIDs (NCT01478022).
            Additionally, it has been studied for its potential use in managing
            chronic periodontitis through subgingival irrigation with a 2% ibuprofen
            mouthwash, which reduces periodontal pocket depth and
            bleeding on probing, improving periodontal health (NCT02538237).

            These findings highlight ibuprofen's versatility in treating pain, inflammation,
            fever, and specific conditions like tension headaches, postoperative pain, and periodontal diseases.

# Automatic Chemistry Descriptors for organic molecules and materials

From a chemist’s standpoint, AutoChemDescriptors is the missing bridge between raw SMILES lists and ready-to-interpret chemical insight. It translates molecular strings into quantitative descriptors, optionally optimizes geometries quantum-mechanically, and produces exploratory analyses that reveal similarities, trends, and outliers across organic molecules and materials—all without micromanaging multiple standalone tools.

## Capabilities at a glance
- Generate RDKit-based descriptors directly from SMILES or invoke PySCF + DScribe to obtain MBTR descriptors informed by electronic-structure optimization.
- Scale calculations across as many processes (`n_jobs`) as your workstation allows, accelerating large virtual libraries.
- Obtain PCA-based grouping, heatmap, and biplot visualizations immediately after descriptor generation.
- Keep audit-ready records (molecule depictions, XYZ snapshots, software versions, execution times) for reproducibility and reporting.
- Run a dedicated K-Means clustering pipeline with elbow/silhouette diagnostics, PCA-based scatter plots, and export-ready reports (Markdown + CSV/JSON) for exploratory grouping.
- Perform density-based clustering with DBSCAN, including automatic eps estimation from the k-distance curve, Tanimoto-distance diagnostics, and publication-quality plots for core/border/noise inspection.

## Before you begin
1. Set up a Python 3 environment and install the packages listed in `requirements.txt` (RDKit, PySCF, DScribe, ASE, NumPy, SciPy, Matplotlib, scikit-learn, Torch, PyBERNY).
2. Place the `auto_chem_descriptors` package somewhere in your project workspace. You are free to organize your runs in separate folders; this guide does not rely on the bundled `examples` directory.

## Preparing your run script
- Create a working directory (for instance, `~/chem_runs/run001`) and inside it create a file named `autochemdesc.py` with the following structure. All parameters—descriptor type, molecule list, calculator settings, PCA configuration, number of jobs, debug flag—are user-editable, but keep the structural skeleton intact.
- edit ~/.bashrc and insert: " export "PYTHONPATH=/mnt/d/workspace/AutoChemDescriptors.0.0.8:$PYTHONPATH" "
- source ~/.bashrch


```python
#from main_auto_chem_descriptor import main_auto_chem_descriptor
from auto_chem_descriptors.main.main_auto_chem_descriptors import main_auto_chem_descriptors

if __name__ == '__main__':

    is_debug_true = False

    n_jobs = 2

    input_flow_controller = {
            'molecular_encoding': "SMILES",
            'descriptors_type': "SMILES",
    }

    molecules_coded_list = [
                 "COc1ccccc1O",
                 "COc1ccccc1OCC(=O)O",
                 "COc1ccccc1OCC(=O)Nc1ccccc1",
                 "COc1ccccc1OCC(=O)Nc1ccccc1F",
                 "COc1ccccc1OCC(=O)Nc1cccc(F)c1",
                 "COc1ccccc1OCC(=O)Nc1ccc(F)cc1",
                 "COc1ccccc1OCC(=O)Nc1cccc2ccccc12",
                ]

    calculator_controller = {}

    n_components = 4

    analysis = {

        "pca_heatmap": [True, n_components],
        "pca_grouping": [True, n_components],
        "pca_dispersion": [True, n_components],

        "kmeans": {
            "k_min": 2,
            "k_max": 8,
            "random_state": 42,
            "use_minibatch": False,
            "projection_components": 2
        },

        "dbscan": {
            "min_samples": 8,
            "metric_mode": "auto",
            "precomputed_max_samples": 1200,
            "eps": 0.35,              # optional; omit to adopt the knee suggestion
            "n_jobs": -1,             # optional; set None to use scikit-learn default
            "algorithm": "brute"      # optional; accepts {'auto','ball_tree','kd_tree','brute'}
        },

        "kmeans_report": {
            "report_filename": "report_kmeans.md",
            "metrics_filename": "kmeans_metrics.csv",
            "suggestions_filename": "kmeans_suggestions.json",
            "labels_filename": "kmeans_cluster_labels.csv"
        },

        "molecules_color": ['b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y',
                            'b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y'],

        "molecules_label": [
          'Guaiacol', 'GACO1', 'GAO00', 'GAA21',
          'GAA31', 'GAA41', 'GAA22']
     }

    main_auto_chem_descriptors(n_jobs,
                              input_flow_controller,
                              molecules_coded_list,
                              calculator_controller,
                              analysis)
```

Feel free to switch `descriptors_type` to `"MBTR"`, add basis-set parameters to `calculator_controller`, or tailor the PCA options; the structure remains identical, and every field is configurable.

### Writing your own descriptor script (what's mandatory and what's optional)
If you prefer a complete end-to-end reference, open `examples/patricia_project07/autochemdesc.py`. The example showcases every dial available, but you rarely need them all. The checklist below distills what the final user must keep and what can be pruned when crafting a custom descriptor script.

```python
from auto_chem_descriptors.main.main_auto_chem_descriptors import main_auto_chem_descriptors

def run_descriptor():
    n_jobs = 2  # mandatory: parallel workers for descriptor generation

    input_flow_controller = {
        "molecular_encoding": "SMILES",  # mandatory: {"SMILES","SELFIES"}
        "descriptors_type": "SMILES",    # mandatory: {"SMILES","MBTR"}
    }

    molecules_coded_list = ["COc1ccccc1O"]  # mandatory: list of encodings

    calculator_controller = {}  # optional: e.g., PySCF, basis set, MBTR grid

    analysis = {  # optional: drop sections you do not need
        "pca_heatmap": [True, 4],     # optional: bool flag + n_components
        "pca_grouping": [False, 3],   # optional: disable by setting False
        "kmeans": {                   # optional: only include if clustering
            "k_min": 2, "k_max": 6,   # required fields inside the block
            "random_state": 42,
        },
        "dbscan": {                   # optional: enable density clustering
            "min_samples": 8,         # mandatory inside DBSCAN block
            "eps": 0.25,              # optional: omit to rely on knee detection
        },
        "molecules_color": ["b"],     # optional: colormap for plots
        "molecules_label": ["Seed"],  # optional: annotations per molecule
    }

    main_auto_chem_descriptors(
        n_jobs,
        input_flow_controller,
        molecules_coded_list,
        calculator_controller,
        analysis,
    )

if __name__ == "__main__":
    run_descriptor()
```

- **Imports and path handling (mandatory):** Ensure `auto_chem_descriptors` is in `PYTHONPATH` or prepend the repo root manually, just as the Patricia example does before importing `main_auto_chem_descriptors`.
- **Execution guard (mandatory):** Keep the `if __name__ == "__main__":` block so you can import the script elsewhere without triggering a run immediately.
- **`input_flow_controller` (mandatory):** Supply both `molecular_encoding` (SMILES or SELFIES) and `descriptors_type` (SMILES or MBTR). AutoChemDescriptors derives downstream behavior from these two keys, so omitting either prevents execution.
- **`molecules_coded_list` (mandatory):** Provide every structure you want in the batch. At least one entry is required; additional strings simply extend the dataset.
- **`calculator_controller` (optional unless you run MBTR):** Leave it empty for pure RDKit descriptors, or define PySCF/DScribe settings (basis set, convergence, grid) when requesting MBTR so quantum and descriptor backends stay synchronized.
- **`analysis` (fully optional):** Remove the dictionary entirely for headless descriptor generation or include only the subsections you need.
  - `pca_heatmap`, `pca_grouping`, and `pca_dispersion` accept `[flag, n_components]`; the list structure is required only when the flag is `True`.
  - `kmeans` requires `k_min`, `k_max`, and `random_state`; `use_minibatch`, `projection_components`, and reporting filenames are optional extras.
  - `dbscan` needs `min_samples`; `eps`, `n_jobs`, `metric_mode`, `precomputed_max_samples`, and `algorithm` tailor runtime versus accuracy and can be dropped when defaults suffice.
  - `molecules_color` and `molecules_label` only matter when you want persistent color/label assignments in plots and reports.
- **`is_debug_true` (optional):** Toggle when you want verbose logs; you can omit it entirely if you prefer the default `False`.
- **`main_auto_chem_descriptors` call (mandatory):** Pass arguments in the order shown (or as keyword arguments). Leaving any of the first four positional parameters undefined will raise validation errors before any calculation starts.

This structure keeps descriptor scripts approachable for end users: start with the mandatory trio (`n_jobs`, `input_flow_controller`, `molecules_coded_list`), add quantum or clustering blocks only when they add value, and rely on the Patricia project script solely as a reference for advanced combinations.

## Running the workflow
Inside the directory that contains `autochemdesc.py`, point `PYTHONPATH` to wherever `auto_chem_descriptors` resides and execute the script. Example session (path names are illustrative):

```
(venv) user@labhost ~/chem_runs/run001 $ PYTHONPATH="../auto_chem_descriptors" python autochemdesc.py
```

The program will report descriptor values, save any requested PNG/XYZ/PCA files, print software version info, and summarize the elapsed time.

## Interpreting the outputs
- **Descriptor arrays** (RDKit or MBTR) can be consumed by clustering, QSAR, or active-learning pipelines.
- **Molecule images and XYZ snapshots** help trace any data point back to its structure or geometry optimization stage.
- **PCA figures** visualize grouping tendencies, feature importances, and dominant directions of variance, providing immediate insight into the sampled chemical space.
- **K-Means artifacts/reports** (`plot_kmeans_elbow.png`, `plot_kmeans_silhouette.png`, `plot_kmeans_clusters.png`, `report_kmeans.md`, `kmeans_metrics.csv`, `kmeans_cluster_labels.csv`, `kmeans_suggestions.json`) complement PCA outputs with partitioning diagnostics, adopted K rationale, and per-sample metadata.
- **DBSCAN artifacts/reports** (`dbscan_parameters.json`, `dbscan_stats.json`, `dbscan_labels.csv`, `plot_dbscan_k_distance.png`, `plot_dbscan_clusters.png`, `report_dbscan.md`) expose density reachability (eps, min_samples, core/border/noise counts) along with k-distance curves, annotated PCA projections, and a Markdown interpreter that contrasts PCA/K-Means/DBSCAN, quantifies epsilon deltas, and appends a decision playbook for future runs.
- **Console logs** register library versions and execution timestamps for reproducibility.

## Configuring K-Means reports
- Reports are generated automatically when `analysis["kmeans"]` is enabled. Customize filenames or integrate with downstream tools by providing an optional `analysis["kmeans_report"]` dictionary (defaults shown in the run-script example).
- The Markdown file (`report_kmeans.md`) summarizes dataset size, estimator settings, suggested K values, cluster balance, and low-silhouette alerts.
- CSV/JSON sidecars (`kmeans_metrics.csv`, `kmeans_cluster_labels.csv`, `kmeans_suggestions.json`) expose the numeric diagnostics (per-K inertia/silhouette/runtimes, per-sample cluster assignments, and elbow/silhouette recommendations) for further analytics pipelines.

## Configuring DBSCAN analysis
- Enable `analysis["dbscan"]` to trigger the Tanimoto-based DBSCAN routine. Only clustering parameters are required; filenames for outputs are fixed to (`dbscan_parameters.json`, `dbscan_stats.json`, `dbscan_labels.csv`, `plot_dbscan_k_distance.png`, `plot_dbscan_clusters.png`) to keep downstream automation predictable.
- `min_samples` defines the density threshold for core points (default `5`). Set `metric_mode` to `"auto"` to let the orchestrator decide between precomputing the full Tanimoto distance matrix or using an online Jaccard search (binary fingerprints only). Force `"precomputed"` if you want the generalized Tanimoto to run even on continuous descriptors; force `"jaccard"` when you know the matrix is binary and small enough.
- If `eps` is omitted, the module computes the k-distance curve (with `k = min_samples`) and adopts the knee point as the density radius. Supplying `eps` overrides that value, while still logging both the suggestion and the user-provided number in `dbscan_parameters.json`.
- Optional knobs: `precomputed_max_samples` (default `1500`) to cap when `metric_mode="auto"` switches to the streaming neighbor approach, `n_jobs` to parallelize brute-force searches, and `algorithm` to choose among scikit-learn’s DBSCAN backends.
- `analysis["dbscan_report"]` (optional dict) lets you rename the Markdown file (`report_filename`, default `report_dbscan.md`) and cap how many entries appear in the noise/border/tail tables (`noise_limit`, `border_limit`, `sparse_limit`). The report always includes the density interpreter (PCA × K-Means × DBSCAN narrative, epsilon discrepancy, chemoinformatics notes) plus a generic decision playbook so each run documents both specific findings and reusable guidance.

## Support and citation
AutoChemDescriptors was created by Maicon P. Lourenço and Clayton V. F. Filho (from GAMIAC research group at the Federal University of Espírito Santo, in Alegre, Brazil). This research is done in collaboration with Dennis R. Salahub (University of Calgary), Jiri Hostas (NRC Canada), Hélio Anderson Duarte and Ângelo de Fátima from Federal University of Mians Gerais. Undergrad students from UFES involved: Stephano Dorigo Scaramussa and Alehando Lopes Gamas. Acknowledgements: REDNIU INCT, FAPES, CNPQ, CAPES. Please, cite the references printed by `software_information_auto_chem_descriptors` in any publication using these workflows. For scientific or technical inquiries, reach out via maiconpl01@gmail.com.***

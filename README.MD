# Automatic Chemistry Descriptors for organic molecules and materials

From a chemist’s standpoint, AutoChemDescriptors is the missing bridge between raw SMILES lists and ready-to-interpret chemical insight. It translates molecular strings into quantitative descriptors, optionally optimizes geometries quantum-mechanically, and produces exploratory analyses that reveal similarities, trends, and outliers across organic molecules and materials—all without micromanaging multiple standalone tools.

## Capabilities at a glance
- Generate RDKit-based descriptors directly from SMILES or invoke PySCF + DScribe to obtain MBTR descriptors informed by electronic-structure optimization.
- Scale calculations across as many processes (`n_jobs`) as your workstation allows, accelerating large virtual libraries.
- Obtain PCA-based grouping, heatmap, and biplot visualizations immediately after descriptor generation.
- Keep audit-ready records (molecule depictions, XYZ snapshots, software versions, execution times) for reproducibility and reporting.

## Before you begin
1. Set up a Python 3 environment and install the packages listed in `requirements.txt` (RDKit, PySCF, DScribe, ASE, NumPy, SciPy, Matplotlib, scikit-learn, Torch, PyBERNY).
2. Place the `auto_chem_descriptors` package somewhere in your project workspace. You are free to organize your runs in separate folders; this guide does not rely on the bundled `examples` directory.

## Preparing your run script
Create a working directory (for instance, `~/chem_runs/run001`) and inside it create a file named `autochemdesc.py` with the following structure. All parameters—descriptor type, molecule list, calculator settings, PCA configuration, number of jobs, debug flag—are user-editable, but keep the structural skeleton intact.

```python
from main_auto_chem_descriptor import main_auto_chem_descriptor

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
                 "COc1ccccc1OCC(=O)Nc1ccccc1Cl",
                 "COc1ccccc1OCC(=O)Nc1cccc(Cl)c1",
                 "COc1ccccc1OCC(=O)Nc1ccc(Cl)cc1",
                 "COc1ccccc1OCC(=O)Nc1ccccc1Br",
                 "COc1ccccc1OCC(=O)Nc1cccc(Br)c1",
                 "COc1ccccc1OCC(=O)Nc1ccc(Br)cc1",
                 "COc1ccccc1OCC(=O)Nc1ccccc1[N+](=O)[O-]",
                 "COc1ccccc1OCC(=O)Nc1cccc([N+](=O)[O-])c1",
                 "COc1ccccc1OCC(=O)Nc1ccc([N+](=O)[O-])cc1",
                 "COc1ccccc1OCC(=O)Nc1ccccc1C",
                 "COc1ccccc1OCC(=O)Nc1cccc(C)c1",
                 "COc1ccccc1OCC(=O)Nc1cccc(C)c1",
                 "COc1ccccc1OCC(=O)Nc1cccc2ccccc12",
                ]

    calculator_controller = {}

    n_components = 4

    analysis = {

        "pca_heatmap": [True, n_components],
        "pca_grouping": [True, n_components],
        "pca_dispersion": [True, n_components],

        "molecules_color": ['b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y',
                            'b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y'],

        "molecules_label": [
          'Guaiacol', 'GACO1', 'GAO00',
          'GAA21', 'GAA31', 'GAA41',
          'GAA22', 'GAA32', 'GAA42',
          'GAA23', 'GAA33', 'GAA43',
          'GAA24', 'GAA34', 'GAA44',
          'GAA25', 'GAA35', 'GAA45', 'GANT']
     }

    main_auto_chem_descriptor(n_jobs,
                              input_flow_controller,
                              molecules_coded_list,
                              calculator_controller,
                              analysis)
```

Feel free to switch `descriptors_type` to `"MBTR"`, add basis-set parameters to `calculator_controller`, or tailor the PCA options; the structure remains identical, and every field is configurable.

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
- **Console logs** register library versions and execution timestamps for reproducibility.

## Support and citation
AutoChemDescriptors was created by Maicon P. Lourenço and Clayton Vieira Fraga Filho (at GAMIAC grou from Federal University of Espírito Santo, in Alegre), from chemistry and computer science background, respectively. This research is done in collaboration with Dennis R. Salahub (University of Calgary) and Jiri Hostas (NRC Canada). Please, Cite the references printed by `software_information_auto_chem_descriptors` in any publication using these workflows. For scientific or technical inquiries, reach out via maiconpl01@gmail.com.***

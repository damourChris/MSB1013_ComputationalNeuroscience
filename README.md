<h1> Multi-layered neural input simulation-based inference 
</h1>


<div align="end" style="display: flex; flex-direction: column; gap: 0.25rem">
    <div>
        <a href="https://damourchris.github.io/MSB1013_ComputationalNeuroscience/">
            <img alt="Docs" src="https://img.shields.io/badge/docs-stable-green">
        </a>
        <a href="https://github.com/damourChris/MSB1013_ComputationalNeuroscience/blob/master/LICENSE">
            <img alt="Github License" src="https://img.shields.io/github/license/damourChris/MSB1013_ComputationalNeuroscience">
        </a>
    </div>


<div>Computational Neuroscience - MSB1013</div>
<div>Systems Biology</div>
<div>Maastricht University</div>
<div>2023</div>

</div>

This project aims to infer the different layers that were given as input to a neuronal model. The output of the model is then transformed into a BOLD model. Finally, the [SBI toolbox](https://sbi-dev.github.io/sbi/) is used to infer the different simulation parameters.

---

<h2>Table of contents</h2>

- [Project](#project)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Project 

The project is divided intro three modules: 
1. Neuron simulations, where the initial training data is generated
2. BOLD Signal, where the data from the neuron simulation is transformed into a BOLD signal 
3. Neural Net inference, where a neural net is used to perform inference to extract to original neuron simulation parameters. 



## Installation


This project uses Python and pip. Go check them out if you don't have them locally installed.

```sh
$ python --version
$ pip --version
```

To install the project, you can clone it locally and install the required packages:
```sh
$ git clone https://github.com/damourChris/MSB1013_ComputationalNeuroscience.git
$ cd MSB1013_ComputationalNeuroscience
$ pip install -r requirements.txt
```
We recommend using a conda environment for managing the dependencies. If you don't have conda installed, you can get it from here. Once you have conda installed, create a new environment and install the required packages:
```sh
$ conda create --name myenv
$ conda activate myenv
$ pip install -r requirements.txt
```

## Contributing
This project originates from the MSB1013 - Computational Neuroscience course of 2023. The codebase is not under active development, however feel free to fork this repo and work on it. Pull request are warmly welcomed. 
*Note: if you find some errors in the documentation, feel free to open up an issue or a PR.*

## License
The source code and documentation are released under the MIT License. See the LICENSE file for more details.
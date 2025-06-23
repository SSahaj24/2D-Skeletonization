# The 2D Skeletonisation Algorithm

## Setting up the pipeline:

- **Conda environment**:
  
  To get started, use the following command to install the necessary libraries and set up a Conda environment tailored to run the package, ensuring all dependencies are met..
  
  `# conda env create -f conda_env.yml`

  Once the previous command has finished executing, activate the environment by running the following command:

  `# conda activate dm++`

  ( Note: Please ensure that conda is already installed; if not, follow the installation instructions provided
  here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html )

- **Compiling C++ code**:

  To compile your C++ code, follow these commands sequentially:

  *Dipha Persistence Module of DM_2D*:
  ```
  # cd DM_2D_code/DiMo2d/code/dipha-2d-thresh/
  # mkdir build
  # cd build
  # cmake ../
  # make
  ```

  *Discrete Morse Graph Reconstruction Module of DM_2D*
  ```
  # cd DM_2D_code/DiMo2d/code/dipha-output-2d-ve-et-thresh/
  # g++ ComputeGraphReconstruction.cpp
  ```
  
- **Kakadu library**:

  Please make sure that the kakadu library is installed. (https://en.wikipedia.org/wiki/Kakadu_(software) )

  Install KAKADU (copy from source v8_2_1-02038E,. if available)

  Kakadu may need these prerequisites:
  
  `sudo apt install make gcc g++ libnuma-dev default-jdk`

  May need to configure Java:
  
  `update-alternatives --config java`
  `java --version`

  Might need to add this line and log out (so Kakadu can find Java).
  
  `#/etc/environment
  JAVA_HOME=/usr/lib/jvm/default-java`

  Test with:
    `printenv JAVA_HOME`

  Extract the zip (copied/downloaded) to:
  ```
  /usr/local/kakadu
  cd v8_2_1-02038E/make
  make -f Makefile-Linux-x86-64-gcc all
  ```
  
  After it is compiled, copy the files:
  ```
  cd ..
  cp lib/Linux-x86-64-gcc/* /usr/lib/
  cp bin/Linux-x86-64-gcc/* /usr/local/bin/
  ```

  Now it should be available to all users on the system. Test with:

  `kdu_compress -v`


## Running the Pipeline (example):

- **For full pipeline WSI images**:
  
  `# python whole_brain_samik_processDet_skel_singleChannel.py example_Input/WSI/ Example_output/WSI/`

  *(python whole_brain_samik_processDet_skel_singleChannel.py {input directory with JP2 file} {output directory})*

    This is used to run the full Pipeline while producing the Mask of the skeletons.

  `# python whole_brain_samik_processDet_skel_singleChannel_lkl.py example_Input /WSI/ Example_output/WSI/`

  *(python whole_brain_samik_processDet_skel_singleChannel_lkl.py {input directory with JP2 files} {output directory})*

    This is used to output the likelihood image only

- **For full pipeline STP images**:

  `# python whole_brain_samik_processDet_skel_gray_STP.py example_Input/STP/ Example_output/STP/`

  *(python whole_brain_samik_processDet_skel_gray_STP.py {input directory with TIF files} {output directory})*

    This is used to run the full Pipeline while producing the Mask of the skeletons.

  `# python whole_brain_samik_processDet_skel_gray_STP_lkl.py example_Input /STP/ Example_output/STP/`

  *(python whole_brain_samik_processDet_skel_gray_STP_lkl.py {input directory with TIF files} {output directory})*

    This is used to output the likelihood image only

- **For standalone images/Tiles from likelihood to skeletons**:

  `# python rundm2d.py lkl/PMD1211_58_5001_12501.tif`

  *(python rundm2d.py {input likelihood TIF image path}*

  `# python plot.json lkl/PMD1211_58_5001_12501.tif json_out/merged_geojson.json image_out/overlayImg.jpg image_out/noOverlayImg.tif`

  *(python plot.json {input likelihood TIF image path} {outputted geojson from rumdm2d.py} {output skelton overlaid image path} {output skeletal image path})*

  
  #### Assumptions

  - If you do not have a Kakadu License, USe openCV in the code as required.
  - The code assumes the WSI image has Signal in the First Channel.
  - An example of each image is given in the [Skeletonization resources page](https://data.brainarchitectureproject.org/pages/skeletonization).


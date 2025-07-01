<h1>FoldCraft</h1>

<h3>Fold-conditioned de novo binder design</h3>
FoldCraft enables fold-conditioning of binder structure, enabling design of binders with diverse folds like TIM-barrels, solenoid folds or Ig-like domains. 
Using VHH conditioned framework FoldCraft can succesfully design single domain nanobody binders against diverse tergets. The FoldCraft pipeline is described in this preprint

You can easily run FoldCraft on Google Colab using this <a href="https://colab.research.google.com/github/KhondamirRustamov/FoldCraft/blob/main/FoldCraft.ipynb">link</a>. Or use this <a href='https://colab.research.google.com/github/KhondamirRustamov/FoldCraft/blob/main/FoldCraft_VHH.ipynb'>notebook</a>, if you want to design VHH fold binders specifically
<br>
<br>


![Fig1-1](https://github.com/user-attachments/assets/b7612207-be45-410d-aaff-fc2586ea765e)


<h2>Installation</h2>

First you need to install FoldCraft repository on your local machine:

`git clone https://github.com/KhondamirRustamov/FoldCraft`

Then run code below to download all requirements, ColabDesign and AlphaFold weights

`bash install_bindcraft.sh --cuda '12.4' --pkg_manager 'conda'`

NOTE: AlphaFold3, which has been used in the manuscript for VHH design benchmarking, should be installed separately as described in official repository: 

<h2>Running FoldCraft locally</h2>

To run FoldCraft locally you first need to download it

**Credits**

This repository uses code from:

* Sergey Ovchinnikov's ColabDesign (https://github.com/sokrypton/ColabDesign)

* Justas Dauparas's ProteinMPNN (https://github.com/dauparas/ProteinMPNN)

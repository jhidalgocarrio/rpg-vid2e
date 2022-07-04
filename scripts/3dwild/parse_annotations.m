load('/media/dani/data/3dwild/mpi_inf_3dhp/S2/Seq1/annot.mat');
dlmwrite("/media/dani/data/3dwild/mpi_inf_3dhp/S2/Seq1/imageSequence/vicon_0.txt", annot2{1,1}, 'delimiter', ' ');
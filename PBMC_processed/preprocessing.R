library(Matrix)
library(biomaRt)

lib_normalize_data <- function (A) {
  #  library normalize a matrix
  
  totalUMIPerCell <- rowSums(A);
  if (any(totalUMIPerCell == 0)) {
    toRemove <- which(totalUMIPerCell == 0)
    A <- A[-toRemove,]
    totalUMIPerCell <- totalUMIPerCell[-toRemove]
    cat(sprintf("Removed %d cells which did not express any genes\n", length(toRemove)))
  }
  
  A_norm <- sweep(A, 1, totalUMIPerCell, '/');
}

if ( !file.exists('./data/purified_pbmc.RData')) {
  urls <- list(b_cells='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz', 
               cd14_monocytes='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd14_monocytes/cd14_monocytes_filtered_gene_bc_matrices.tar.gz',
               cd34='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd34/cd34_filtered_gene_bc_matrices.tar.gz',
               cd4_helper='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd4_t_helper/cd4_t_helper_filtered_gene_bc_matrices.tar.gz',
               regulatory_t='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/regulatory_t/regulatory_t_filtered_gene_bc_matrices.tar.gz',
               naive_t='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_t/naive_t_filtered_gene_bc_matrices.tar.gz',
               memory_t='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/memory_t/memory_t_filtered_gene_bc_matrices.tar.gz',
               cd56_nk='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd56_nk/cd56_nk_filtered_gene_bc_matrices.tar.gz',
               cytotoxic_t='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cytotoxic_t/cytotoxic_t_filtered_gene_bc_matrices.tar.gz',
               naive_cytotoxic='http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_cytotoxic/naive_cytotoxic_filtered_gene_bc_matrices.tar.gz'	)
  
  
  A <- c()
  labels = c()
  for (i in seq_along(urls)) {
    print(i)
    label <-names(urls)[i]
    fn <- sprintf('./data/purified_pbmcs/%s.tar.gz', label)
    download.file(urls[[i]],fn)
    # untar
    fn2 <- sprintf('./data/purified_pbmcs/%s_unzipped' ,label)
    untar(fn, exdir=fn2)
    mtx <- as.matrix((readMM(
      sprintf('%s/filtered_matrices_mex/hg19/matrix.mtx', fn2))))
    genenames <- read.delim(sprintf('%s/filtered_matrices_mex/hg19/genes.tsv', fn2),
                            sep = '\t',header = FALSE)[,2]
    rownames(mtx) <- genenames
    if (i>1 && !all(rownames(mtx) == colnames(A))) {
      error('Trying to concatenate a matrix with different genenames')
    }
    A <- rbind(A,t(mtx))
    labels <- c(labels, rep(label,ncol(mtx)))
  }
  save(A,labels,file='./data/purified_pbmc.RData')
}else{
  print("Loading");
  load('./data/purified_pbmc.RData')
}


num_of_genes_in_cell <- rowSums(A>0)
num_of_cells_in_gene <- colSums(A>0)
keep_cells = which(num_of_genes_in_cell > 400) 
keep_genes = which(num_of_cells_in_gene > 100)

A_ <- A[keep_cells,keep_genes]
cell_labels <- as.factor(labels[keep_cells])
libnorm_data <- lib_normalize_data(A_)

#keep protein coding genes
ensembl=useMart("ensembl")
ensembl=useDataset("hsapiens_gene_ensembl",mart=ensembl)
genes.with.id=getBM(attributes=c("external_gene_name"),filters='biotype', values=c('protein_coding'), mart= ensembl)
reference_genes <- genes.with.id$external_gene_name[14:length(genes.with.id$external_gene_name)]

original_genes <- colnames(libnorm_data)
keep_genes <- original_genes[original_genes %in% reference_genes]
libnorm_data<- libnorm_data[,keep_genes]

# only t cell types
celltypes2keep = c("memory_t", "naive_t", "regulatory_t", "naive_cytotoxic")
cell2keep_ind = cell_labels %in% celltypes2keep
tcells_data <- libnorm_data[cell2keep_ind,]
tcells_labels <- cell_labels[cell2keep_ind]
table(tcells_labels)
# take 90% of the data to select 2k variable genes, then 10% as input to lspin
set.seed(34)
trainset_ind <- sample(x = 1:nrow(tcells_data),size=(0.9*nrow(tcells_data)))
testset_ind <- c(1:nrow(tcells_data))[!(c(1:nrow(tcells_data)) %in% trainset_ind)]
gene_vars <- matrixStats::colVars(tcells_data[trainset_ind,])
names(gene_vars) <- colnames(tcells_data)
gene_vars_sorted <- sort(gene_vars,decreasing = T)
# take the top 2k var genes
var2kgenes <- names(gene_vars_sorted)[1:2000]
remain_data <- scale(tcells_data[testset_ind,var2kgenes])
remain_cell_labels <- tcells_labels[testset_ind]
write.csv(remain_cell_labels,file="./cell_labels_t_vardata.csv",quote=FALSE,sep = ",",row.names = FALSE,col.names = FALSE)
write.csv(remain_data,file="./libnorm_t_vardata.csv",quote = FALSE)




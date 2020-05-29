#########################################################################
#
# Auxiliar program to run all experiments in parallel
#
# Author: Marcos A. Domingues
# Date: September, 2019
#
#########################################################################

# Setup the input and output dir to run all experiments at the same time
runExperiments <- function(from='/home/mad/Experimentos/mad/CARSlibrary/topics', to='/home/mad/Experimentos/mad/CARSlibrary'){
  
  files <- list.files(from)
  
  for(i in files){
    
    file.copy(paste(from, '/', i, sep=''), paste(to, '/', 'dataset.csv', sep=''))

    source(file="daviBEST.R")
    source(file="filterPoF.R")
    source(file="ibcf.R")
    source(file="cReduction.R")
    source(file="weightPoF.R")
              
    tmp <- unlist(strsplit(as.character(i), "\\."))[1]
    
    file.rename("/home/mad/Experimentos/mad/CARSlibrary/result", paste(to, '/', tmp, sep=''))
    
    dir.create("/home/mad/Experimentos/mad/CARSlibrary/result")
    
    file.remove(paste(to, '/', 'dataset.csv', sep=''))
    
  }
  
}

runExperiments()


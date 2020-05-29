#########################################################################
#
# DaVI-BEST Algorithm using Item-Based Collaborative Filtering
#
# Author: Marcos A. Domingues
# Date: September, 2019
#
#########################################################################

###################################################
# Before running these algorithms, please, install the doParallel package
# install.package('doParallel')
###################################################

library(doParallel)
# registerDoParallel(cores=detectCores())
registerDoParallel(cores=10) # Setup number of cores to be used by the cars algorithms

model.name <<- 'tmp_daviBEST.txt'

# DaVI-BEST algorithm for 10-fold cross validation in parallel
davi.run.all.folders.parallel <- function(dataFile='dataset.csv', neigh=4){

	foreach(i=1:10) %dopar% {
		model.name <<- paste('tmp_daviBEST_', i, '.txt', sep='')
		resultFile <- paste('result_dav_', i, '.txt', sep='')
		davi.alg(dataFile,resultFile,neigh,i)
		folder.name <- paste('dav', i, sep='')
		dir.create(paste('result/',folder.name,sep=''))
		file.rename(resultFile,paste('result/',folder.name,'/result_4.csv',sep=''))
	}
}
	
# Run the DaVI-BEST algorithm using 10-fold cross validation
davi.run.all.folders <- function(dataFile='dataset.csv', resultFile='result_dav_4.csv', neigh=4){
	folders <- c(1,2,3,4,5,6,7,8,9,10)

	for(i in folders){
		davi.alg(dataFile,resultFile,neigh,i)
		folder.name <- paste('dav', i, sep='')
		dir.create(paste('result/',folder.name,sep=''))

		file.rename(resultFile,paste('result/',folder.name,'/result_4.csv',sep=''))
	}
}

# Run the DaVI-BEST algorithm for only one fold
davi.run.one.fold <- function(dataFile='dataset.csv', resultFile='result', neigh=c(4), fold=1){
#	idneigh <- c(2,3,4)
	idneigh <- neigh
	for(i in idneigh){
		resultFile <- paste(resultFile,'_',i,'.csv',sep='')
		davi.alg(dataFile,resultFile,i,fold)
	}
}

# The DaVI-BEST algorithm
davi.alg <- function(dataFile='dataset.csv', resultFile='result_4.csv', neigh=4, fold=1){
	cat("...Running the experiments...\n")

	res_pre <- NULL
	res_pre <- read.csv(dataFile, header = TRUE, sep = " ")
	dimensions.names <<- names(res_pre)
	data <- davi.load.data.and.setup(res_pre, fold, 10)

	# Offline step
	cat("...DaVI Offline...\n")
	best.dims <- davi.offline(data, neigh)
	
	# Online step
	cat("...DaVI Online...\n")
	davi.online(data, resultFile, neigh, best.dims)

	cat("...Experiments finished...\n")
}

# Offline step of the algorithm DaVI-BEST algorithm, that will identify the best dimension
davi.offline <- function(data, neigh){

	# Select the best dimensions using 5 fold cross validation
	folders <- c(1,2,3,4,5)
	result.all.folders <- NULL

	for(f in folders){
		data2validation <- davi.data.setup(data$train, f, 5)

		result <- NULL
		result <- davi.cf.exp(Train=unique(data2validation$train[,c(1,4)]), Test=unique(data2validation$test[,c(1,4)]), Test.ids=data2validation$test.ids, Hidden=data2validation$hidden, Table=names(data2validation$train[4]), Field=names(data2validation$train[4]), Ntop=c(1,2,3,5,10), Neib=neigh)

		# Run DaVI approach for each dimension
		for(i in 5:ncol(data2validation$train)) {
			# Apply DaVI approach on the training set
			dim.train <- data.frame(sesId=c(rep(data2validation$train[,1], each=2)), urlId=c(apply(data2validation$train, 1, function(r) {r[c(i,4)]})))
			#dim.train <- dim.train[dim.train[,2]!='d:null',]
			dim.train <- unique(dim.train[,c(1,2)])

			# Apply DaVI approach on the test set
			dim.test <- data.frame(sesId=c(rep(data2validation$test[,1], each=2)), urlId=c(apply(data2validation$test, 1, function(r) {r[c(i,4)]})))
			#dim.test <- dim.test[dim.test[,2]!='d:null',]
			dim.test <- unique(dim.test[,c(1,2)])
	
			result_tmp <- NULL
			result_tmp <- davi.cf.exp(Train=dim.train, Test=dim.test, Test.ids=data2validation$test.ids, Hidden=data2validation$hidden, Table=names(data2validation$train[i]), Field=names(data2validation$train[i]), Ntop=c(1,2,3,5,10), Neib=neigh)
			result <- rbind(result, result_tmp)
		}

		result.all.folders <- rbind(result.all.folders, result)
	}
	best.dimensions <- davi.get.best.dimension(result.all.folders, as.character(names(data$train[4])))
	best.dimensions
}

# Compare and obtain the best dimension
davi.get.best.dimension <- function(results, item.name){

	dim <- as.vector(unique(results[,1]))
	top <- as.vector(unique(results[,3]))

	# Compare F1 values for each dimension and top-N against the user x item model
	final.result <- NULL
	for(i in dim) {      
		for(j in top) { 

			if(i != item.name){
				odt <- results[results[,1] == i & results[,3] == j,]
				odtOnly_items <- results[results[,1] == item.name & results[,3] == j,]

				pvalue <- t.test(as.numeric(odt[,9]), as.numeric(odtOnly_items[,9]), paired=T, alternative='greater', conf=0.95)[[3]]
				if( (!is.na(pvalue)) & (pvalue < 0.05) ){
					final.result <- rbind(final.result, data.frame(Dim=i,topN=j,fm1=mean(as.numeric(odt[,9]))))
				}
			}
		}	
	}

	best.dimensions <- NULL
	if(is.null(final.result)){ # No dimensions provide better results
		best.dimensions <- rbind(best.dimensions, data.frame(Dim=rep(item.name, length(top)), topN=top)) #The user x item is selected for each top-N
	} else {
		for(i in top){ # Select the best dimension for each top-N
			f.r.t <- final.result[final.result[,2]==i,]
			if (nrow(f.r.t) > 0){
				d <- f.r.t[which.max(f.r.t[,3]),1]
				best.dimensions <- rbind(best.dimensions, data.frame(Dim=d,topN=i))
			} else {
				best.dimensions <- rbind(best.dimensions, data.frame(Dim=item.name,topN=i))
			}
		}
	}
	best.dimensions # Table with the best dimensions for each top-N
}

davi.cf.exp <- function(Train=NULL, Test=NULL, Test.ids=NULL, Hidden=NULL, Table='', Field='', Ntop=c(1), Neib=2){
	results <- NULL
	S <- davi.simmatrix(Train)
	results <- davi.cf.reeval(S,Test,Test.ids,Hidden,Table,Field,Ntop,Neib)
	results
}

# Take a test set, a hidden set and a similarity matrix
# and outputs evaluation statistics for each topN
davi.cf.reeval <- function(S,Test,Test.Ids,Hid,Table,Field,Ntop,Neib) {
	results <- NULL
	Recs <- davi.topNrec.batch(Test,Test.Ids,S,Ntop,Neib) # Recommendations for each topN
	
	uids <- NULL
	recs <- NULL
	for(i in 1:length(Recs)){
		idx <- Recs[[i]][[1]]
		items <- (Recs[[i]][[2]])[[1]]
		recs <- c(recs,items)
		uids <- c(uids,rep(idx,length(items)))		
	}
	RecsTmp <- data.frame(sesId=uids,urlId=recs)
	res <- davi.rec.eval(RecsTmp,Hid,Test.Ids,S,Test) # Evaluate the recommendations and return statistics
	results <- data.frame(EXP.TABLE=factor(Table, levels=dimensions.names), EXP.FIELD=factor(Field, levels=dimensions.names), N=Ntop[1], HITS=res$hits, N.RECS=res$nrecs, N.HIDDEN=res$nhidden, RECALL=res$recall, PRECISION=res$precision, F1=res$f1, FALLOUT=res$fallout, MAP=res$map)
	
	for(j in 2:length(Ntop)){
		uids <- NULL
		recs <- NULL
		for(i in 1:length(Recs)){
			idx <- Recs[[i]][[1]]
			items <- (Recs[[i]][[2]])[[j]]
			recs <- c(recs,items)
			uids <- c(uids,rep(idx,length(items)))		
		}
		RecsTmp <- data.frame(sesId=uids,urlId=recs)
		res <- davi.rec.eval(RecsTmp,Hid,Test.Ids,S,Test) # Evaluate the recommendations and return statistics
		results <- rbind(results, c(Table,Field,Ntop[j],res$hits,res$nrecs,res$nhidden,res$recall,res$precision,res$f1,res$fallout,res$map))
	}
	results
}

# Gives topN recommendations for each user session
davi.topNrec.batch <- function(Obs,ObsIds,S,N,Neib) {
	Usr <- as.vector(ObsIds)

	allTopNrec <- list()
	idxtopNrec <- 1
	
	for(u in Usr) {
		lrec <- davi.topNrec(Obs[Obs[,1]==u,2],S,N,Neib)
		oneTopNrec <- list(idx=u,recs=lrec)
		allTopNrec[[idxtopNrec]] <- oneTopNrec
		idxtopNrec <- idxtopNrec + 1
	}
	allTopNrec
}

# Gives topN recommendations for one user session
davi.topNrec <- function(R,S,N,Neib) {
	recs <- davi.wsum(R,S,Neib)

	max <- length(recs)
	Nrecs <- list()
	idxNrecs <- 1

	for(i in N) {
		if(max==0) {
			Nrecs[[idxNrecs]] <- c(numeric(0))
			idxNrecs <- idxNrecs + 1
		} else {
			#sort(recs,decreasing=T)[1:min(N,max)]
			my.x <- recs
			Nrecs[[idxNrecs]] <- names(recs[sapply(1:(min(i,max)), function(dummy) {my.max <- which.max(my.x); my.x[my.max] <<- -1; my.max})]) 
			idxNrecs <- idxNrecs + 1
		}
	}
	Nrecs
}

# Retrieve the score for each candidate recommendation for a user session
davi.wsum <- function(R,S,Neib) {
	idx <- names(S)
	sdftemp <- setdiff(idx,R)

	sdf <- sdftemp[grep('^h:', sdftemp)]
	n <- length(sdf)

	B <- sapply(1:n, function(i){ 
		davi.wsumitem(R,S,sdf[i],Neib)
	})
	names(B) <- sdf
	B
}

# Compute the score for one candidate recommendation
davi.wsumitem <- function(R,S,i,neighb) {
	Neighbs <- davi.neighbors(i,S,neighb)
	
	N <- intersect(names(Neighbs),R)
	num <- sum(Neighbs[N])
	den <- sum(Neighbs)

	if (den==0) 0
	else num / den
}

# Find the neighbors
davi.neighbors <- function(i,S,Neib){
	neigh <- c(0)
	names(neigh) <- 'NoNeigh'
	M1 <- S[[i]]

	if(!is.null(M1)){
		max <- length(M1)
		my.xx <- M1
		neigh <- M1[sapply(1:(min(Neib,max)), function(dummy) {my.max <- which.max(my.xx); my.xx[my.max] <<- -1; my.max})] 
	}
	neigh
}

# Evaluate the recommendations and compute the metrics and statistics
davi.rec.eval <- function(Recs, Hid, ObsIds, S, Obs){
	nObs <- length(as.vector(ObsIds))
	hits <- 0
	recall.per.user <- NULL
	precision.per.user <- NULL
	f1.per.user <- NULL
	f.per.user <- NULL
	m.per.user <- NULL
	
	candidates <- names(S)
	candidates <- candidates[grep('^h:', candidates)]

	# Compute metrics for each user
	for(u in as.vector(unique(Recs[,1]))){
		hits_tmp <- 0
		hits_tmp <- davi.hits.per.user(Recs,Hid,u)
		hits <- hits + hits_tmp

		hid_user <- as.vector(Hid[Hid[,1]==u,2])
		recs_user <- as.vector(Recs[Recs[,1]==u,2])
		obs_user <- as.vector(Obs[Obs[,1]==u,2])

		# Compute Recall		
		r <- 0
		r <- hits_tmp/length(hid_user)
		recall.per.user <- c(recall.per.user,r)
		
		# Compute Precision
		p <- 0
		p <- hits_tmp/length(recs_user)
		precision.per.user <- c(precision.per.user,p)
		
		# Compute F1 metric
		f1m <- 0
		if((r != 0) & (p != 0)) {
			f1m <- ((2*r*p)/(r+p))
			f1.per.user <- c(f1.per.user,f1m)
		} else {
			f1.per.user <- c(f1.per.user,f1m)
		}

		# Compute Fallout
		candidates_user <- setdiff(candidates, obs_user)
		f <- 0
		f <- ((length(recs_user)) - (hits_tmp)) / (length(setdiff(candidates_user, hid_user)))
		f.per.user <- c(f.per.user, f)

		# Compute MAP
		m <- 0
		if(hits_tmp == 1){
			m <- hits_tmp / as.numeric(which(recs_user==hid_user))
		}
		m.per.user <- c(m.per.user, m)
	}
	
	OUT <- NULL
	OUT$hits <- hits
	OUT$nrecs <- nrow(Recs)
	OUT$nhidden <- nrow(Hid)
	recall <- sum(recall.per.user)/nObs # Compute average recall
	OUT$recall <- recall
	precision <- sum(precision.per.user)/nObs # Compute average precision
	OUT$precision <- precision
	f1 <- sum(f1.per.user)/nObs # Compute average F1 metric
	OUT$f1 <- f1
	fallout <- sum(f.per.user)/nObs # Compute average fallout
	OUT$fallout <- fallout
	map <- sum(m.per.user)/nObs # Compute average map
	OUT$map <- map
	OUT
}

# Compute intersection between recommendation and ground-truth
davi.hits.per.user <- function(R,H,u) {
	length(intersect(R[R[,1]==u,2],H[H[,1]==u,2]))
}

# Prepare data for the experiments
davi.load.data.and.setup <- function(Dataset, Fold, Nfolds) {
	Dataset[,1] <- as.numeric(Dataset[,1])
	for(i in 4:ncol(Dataset)){
		Dataset[,i] <- as.character(Dataset[,i])
	}
	davi.data.setup(Dataset, Fold, Nfolds)
}

# Prepare data for the experiments and create the folds
davi.data.setup <- function(data, fold, nfolds) {
	OUT <- NULL
	
	data <- data[order(data[,1]),]
	data.ids <- as.vector(unique(data[,1]))
	
	max <- length(data.ids)
	set.seed(20)
	sample <- rep(1:nfolds, length=max)[order(runif(max))]

	train.ids <- data.ids[sample != fold]
	data.train.idx <- NULL
	for(i in train.ids) { 
		data.train.idx <- c(data.train.idx,which(data[,1]==i))
	}
	data.train <- data[data.train.idx,]
		
	test.ids <- data.ids[sample == fold]
	data.test.all <- NULL
	data.test.unique <- NULL
	data.hidd.idx <- NULL
	data.hidd.item <- NULL
	for(i in test.ids){
		data.partition <- data[which(data[,1]==i),]
		
		if(nrow(unique(data.partition[,c(1,4)])) > 1){
			hidsample <- sample(1:nrow(data.partition),1) # random
			# hidsample <- nrow(data.partition) # last
			
			hidd.idx <- data.partition[hidsample,1]
			hidd.item <- data.partition[hidsample,4]
			data.hidd.idx <- c(data.hidd.idx,hidd.idx)
			data.hidd.item <- c(data.hidd.item,hidd.item)
			data.partition <- data.partition[data.partition[,4]!=hidd.item,]
			data.test.all <- rbind(data.test.all,data.partition)
		} else {
			data.test.unique <- c(data.test.unique,i)	
		}
	}
	test.ids.ok <-setdiff(test.ids,data.test.unique)
	data.hidd <- data.frame(sesId=data.hidd.idx, urlId=data.hidd.item)
	
	OUT$train <- data.train
	OUT$test <- data.test.all
	OUT$test.ids <- test.ids.ok
	OUT$hidden <- data.hidd
	OUT
}

# Compute the similarity matrix using adjacency list
davi.simmatrix <- function(A) {
	i.u <- tapply(A[,1], A[,2], function(x){x})
	items <- names(i.u)
	nitems <- length(items)
	
	# correlation similarity
	# triang.m <- lapply(1:(length(i.u)-1), function(ind, i.u) {sapply((ind+1):length(i.u), function(ind2, i.u, ind1) {((length(unique(A[,1]))*length(intersect(i.u[[ind1]], i.u[[ind2]])))-(length(i.u[[ind1]])*length(i.u[[ind2]])))/(sqrt((length(unique(A[,1]))*length(i.u[[ind1]])-(length(i.u[[ind1]])^2))*(length(unique(A[,1]))*length(i.u[[ind2]])-(length(i.u[[ind2]])^2))))}, i.u, ind)}, i.u)
	
	for (j in 1:(nitems-1)){
		item1 <- NULL
		item2 <- NULL
		sim <- NULL
		# cosine similarity
		sim <- sapply((j+1):nitems, function(ind2, i.u, ind1) {(length(intersect(i.u[[ind1]], i.u[[ind2]])))/(sqrt(length(i.u[[ind1]]))*sqrt(length(i.u[[ind2]])))}, i.u, j)
		item2 <- items[(j+1):nitems]
		item1 <- rep(items[j], length(item2))
    		df <- data.frame(item1, item2, sim)
		df <- df[df[,3] > 0,]
		write.table(df, append = TRUE, file = model.name, row.names = FALSE, col.names = FALSE, quote = FALSE, sep = "\t")
		write.table(df[,c(2,1,3)], append = TRUE, file = model.name, row.names = FALSE, col.names = FALSE, quote = FALSE, sep = "\t")
	}

	dd <- read.csv(model.name, header = FALSE, sep = "\t")
	idx <- as.character(unique(dd[,1]))
	sim <- list()
	
	for (i in idx){
		tmp <- dd[dd[,1]==i,c(2,3)]
		tmp1 <- as.vector(tmp[,2])
		names(tmp1) <- as.character(tmp[,1])
		sim[[i]] <- tmp1
	}
	file.remove(model.name)
	sim
}

# Online step of the DaVI-BEST algorithm
davi.online <- function(data, resultFile, neigh, best.dims) {
	write.table(data.frame(EXP.TABLE='EXP.TABLE', EXP.FIELD='EXP.FIELD', N='N', HITS='HITS', N.RECS='N.RECS', N.HIDDEN='N.HIDDEN', RECALL='RECALL', PRECISION='PRECISION', F1='F1', FALLOUT='FALLOUT', MAP='MAP'), file = resultFile, row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")

	item.name <- as.character(names(data$train[4]))

	for(i in 1:nrow(best.dims)){

		if(as.character(best.dims[i,1]) == item.name){

			S <- davi.simmatrix(unique(data$train[,c(1,4)]))
			result <- davi.online.cf.exp(S, unique(data$test[,c(1,4)]), data$test.ids, data$hidden, 'daviBEST', 'ibcf', as.numeric(as.vector(best.dims[i,2])), neigh)
			write.table(result, append = TRUE, file = resultFile, row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")

		} else {

			dim.train <- data.frame(sesId=c(rep(data$train[,1], each=2)), urlId=c(apply(data$train, 1, function(r) {r[c(as.character(best.dims[i,1]), item.name)]})))
			#dim.train <- dim.train[dim.train[,2]!='d:null',]
			dim.train <- unique(dim.train[,c(1,2)])
			dim.test <- data.frame(sesId=c(rep(data$test[,1], each=2)), urlId=c(apply(data$test, 1, function(r) {r[c(as.character(best.dims[i,1]), item.name)]})))
			#dim.test <- dim.test[dim.test[,2]!='d:null',]
			dim.test <- unique(dim.test[,c(1,2)])

			S <- davi.simmatrix(dim.train)
			result <- davi.online.cf.exp(S, dim.test, data$test.ids, data$hidden, 'daviBEST', tolower(as.character(best.dims[i,1])), as.numeric(as.vector(best.dims[i,2])), neigh)
			write.table(result, append = TRUE, file = resultFile, row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")

		}
	}
}

# Compute the online part (recommendations) of the DaVI-BEST algorithm
davi.online.cf.exp <- function(S,Test,Test.Ids,Hid,Table,Field,Ntop,Neib) {
	Recs <- davi.online.topNrec.batch(Test,Test.Ids,S,Ntop,Neib)
	res <- davi.rec.eval(Recs,Hid,Test.Ids,S,Test)
	result <- data.frame(EXP.TABLE=Table, EXP.FIELD=Field, N=Ntop, HITS=res$hits, N.RECS=res$nrecs, N.HIDDEN=res$nhidden, RECALL=res$recall, PRECISION=res$precision, F1=res$f1, FALLOUT=res$fallout, MAP=res$map)
	result
}

# Gives topN recommendations for each user session
davi.online.topNrec.batch <- function(Obs,ObsIds,S,N,Neib) {
	Usr <- as.vector(ObsIds)
	uids <- NULL
	recs <- NULL

	for(u in Usr) {
		urec <- names(davi.online.topNrec(Obs[Obs[,1]==u,2],S,N,Neib))
		recs <- c(recs,urec)
		uids <- c(uids,rep(u,length(urec)))
	}
	data.frame(uids,recs)
}

# Gives topN recommendations
davi.online.topNrec <- function(R,S,N,Neib) {
	recs <- davi.wsum(R,S,Neib)
	max <- length(recs)
	my.x <- recs
	recs[sapply(1:(min(N,max)), function(dummy) {my.max <- which.max(my.x); my.x[my.max] <<- -1; my.max})] 
}

# run all folders in parallel
davi.run.all.folders.parallel()
# run all folders sequentially
# davi.run.all.folders()


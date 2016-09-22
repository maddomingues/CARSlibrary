#########################################################################
#
# Combined Reduction Based Algorithm using Item-Based Collaborative Filtering
#
# Author: Marcos A. Domingues
# Date: September, 2016
#
#########################################################################

###################################################
# Before running these algorithms, please, install the doParallel package
# install.package('doParallel')
###################################################

library(doParallel)
# registerDoParallel(cores=detectCores())
registerDoParallel(cores=10) # Setup number of cores to be used by the cars algorithms

model.name <<- 'tmp_cReduction.txt'
segment.name <<- 'segments'

# cReduction algorithm for 10-fold cross validation in parallel
cReduction.run.all.folders.parallel <- function(dataFile='dataset.csv', neigh=4){

	foreach(i=1:10) %dopar% {
		segment.name <<- paste('segments_', i, sep='')
		dir.create(segment.name)

		model.name <<- paste('tmp_cReduction_', i, '.txt', sep='')
		resultFile <- paste('result_crb_', i, '.txt', sep='')
		cReduction.run.exp.cf(dataFile,resultFile,neigh,i)
		folder.name <- paste('crb', i, sep='')
		dir.create(paste('result/',folder.name,sep=''))

		file.rename(resultFile,paste('result/',folder.name,'/result_4.csv',sep=''))
		file.rename(segment.name,paste('result/',folder.name,'/segments',sep=''))
	}
}
	
# Run the cReduction algorithm using 10-fold cross validation
cReduction.run.all.folders <- function(dataFile='dataset.csv', resultFile='result_crb_4.csv', neigh=4){
	folders <- c(1,2,3,4,5,6,7,8,9,10)

	for(i in folders){
		dir.create(segment.name)
		cReduction.run.exp.cf(dataFile,resultFile,neigh,i)
		folder.name <- paste('crb', i, sep='')
		dir.create(paste('result/',folder.name,sep=''))

		file.rename(resultFile,paste('result/',folder.name,'/result_4.csv',sep=''))
		file.rename(segment.name,paste('result/',folder.name,'/',segment.name,sep=''))

	}
}

# Run the cReduction algorithm for only one fold
cReduction.run.one.fold <- function(dataFile='dataset.csv', resultFile='result', neigh=c(4), fold=1){
#	idneigh <- c(2,3,4)
	idneigh <- neigh
	for(i in idneigh){
		dir.create(segment.name)
		resultFile <- paste(resultFile,'_',i,'.csv',sep='')
		cReduction.run.exp.cf(dataFile,resultFile,i,fold)
	}
}

# The cReduction algorithm - intermediate call
cReduction.run.exp.cf <- function(dataFile='dataset.csv', resultFile='result_4.csv', neigh=4, fold=1) {
	cat("...Running the experiments...\n")

	res_pre <- NULL
	res_pre <- read.csv(dataFile, header = TRUE, sep = " ")

	result <- NULL
	result <- cReduction.cf.exp(Dataset=res_pre, Table='cReduction', Field='cReduction', Ntop=c(1,2,3,5,10), Neib=neigh, Fold=fold)
	write.table(result, file = resultFile, row.names = FALSE, col.names = TRUE, quote = FALSE, sep = ",")

	cat("...Experiments finished...\n")

	result
}

# The cReduction algorithm
cReduction.cf.exp <- function(Dataset=NULL, Table='', Field='', Ntop=c(1), Neib=4, Fold=1) {

	data <- cReduction.load.data.and.setup(Dataset, Fold)
	segments <- cReduction.offline.create.all.segments(data$train,0.2) # Create all segments
	
	S <- cReduction.simmatrix(unique(data$train[,c(1,4)]))

	segmentsC <- cReduction.offline.create.contextual.segments(data$train, segments, Ntop, Neib) # Compute the contextual segments

	result <- NULL
	for(i in 1:length(Ntop)){
		write.segmentsC(data$train,segmentsC[[i]],Ntop[i]) # PRA ANALISE - APAGAR
		res <- cReduction.cf.reeval(S,segmentsC[[i]],data$test,data$test.ids,data$hidden,Ntop[i],Neib)$stats
		result_tmp <- data.frame(EXP.TABLE=Table, EXP.FIELD=Field, N=Ntop[i], HITS=res$hits, N.RECS=res$nrecs, N.HIDDEN=res$nhidden, RECALL=res$recall, PRECISION=res$precision, F1=res$f1, FALLOUT=res$fallout, MAP=res$map)
		result <- rbind(result, result_tmp)
	}

	result
}

# Take a test set, a hidden set and a similarity matrix
# and outputs evaluation statistics for each topN
cReduction.cf.reeval <- function(S,segsC,Test,Test.Ids,Hid,Ntop,Neib) {
	Recs <- cReduction.topNrec.batch(Test,S,segsC,Ntop,Neib) # Recommendations for a topN
  
  # Transform test format to data.frame format
	sesObs <- NULL
	iteObs <- NULL
	for(i in 1:length(Test)){
	  itemsObs <- as.vector(Test[[i]][[2]])
	  iteObs <- c(iteObs, itemsObs)
	  sesObs <- c(sesObs, rep(Test[[i]][[1]], length(itemsObs)))
	}
	frameObs <- data.frame(sesObs, iteObs)
  print(frameObs)
	stats <- cReduction.rec.eval(Recs,Hid,Test.Ids,S,frameObs) # Evaluate the recommendations and return statistics
	OUT <- NULL
	#OUT$recs <- Recs
	OUT$stats <- stats
	OUT
}

# Gives topN recommendations for each user session using the contextual segment or not
cReduction.topNrec.batch <- function(Obs,S,segC,N,Neib) {
	uids <- NULL
	recs <- NULL

	if(length(segC) > 0){ # There exists some segments
		for(u in 1:length(Obs)) {
			ObsItems <- Obs[[u]][[2]]
			ObsDimen <- Obs[[u]][[3]]
			i <- 1
			seg.idx <- 0
			stop <- FALSE
			while((i <= length(segC)) & (stop == FALSE)){
#				if( ((paste(segC[[i]][[2]],segC[[i]][[3]],sep='=') %in% ObsDimen)==TRUE) ){
				if( ((paste(segC[[i]][[2]],segC[[i]][[3]],sep='=') %in% ObsDimen)==TRUE) & (all(ObsItems %in% segC[[i]][[4]])==TRUE)){
					stop <- TRUE
					seg.idx <- i
				}
				i <- i + 1
			}
			if(stop == TRUE){
				print('Using the segment') # Use the contextual segment
				urec <- names(cReduction.topNrec(ObsItems,segC[[seg.idx]][[5]],N,Neib))
				recs <- c(recs,urec)
				uids <- c(uids,rep(Obs[[u]][[1]],length(urec)))
			} else {
				print('Using all data') # Use the the regular ibcf
				urec <- names(cReduction.topNrec(ObsItems,S,N,Neib))
				recs <- c(recs,urec)
				uids <- c(uids,rep(Obs[[u]][[1]],length(urec)))
			}
		}
	} else {
		for(u in 1:length(Obs)) { # There is not segments, than use the regular ibcf
			urec <- names(cReduction.topNrec(Obs[[u]][[2]],S,N,Neib))
			recs <- c(recs,urec)
			uids <- c(uids,rep(Obs[[u]][[1]],length(urec)))
		}
	}
	data.frame(uids,recs)
}

# Gives topN recommendations for one user session
cReduction.topNrec <- function(R,S,N,Neib) {
	recs <- cReduction.wsum(R,S,Neib)
	max <- length(recs)
	my.x <- recs
	recs[sapply(1:(min(N,max)), function(dummy) {my.max <- which.max(my.x); my.x[my.max] <<- -1; my.max})] 
}

# Retrieve the score for each candidate recommendation for a user session
cReduction.wsum <- function(R,S,Neib) {
	idx <- names(S)
	sdf <- setdiff(idx,R)
	n <- length(sdf)

	B <- sapply(1:n, function(i){ 
		cReduction.wsumitem(R,S,sdf[i],Neib)
	})
	names(B) <- sdf
	B
}

# Compute the score for one candidate recommendation
cReduction.wsumitem <- function(R,S,i,neighb) {
	Neighbs <- cReduction.neighbors(i,S,neighb)
	
	N <- intersect(names(Neighbs),R)
	num <- sum(Neighbs[N])
	den <- sum(Neighbs)

	if (den==0) 0
	else num / den
}

# Find the neighbors
cReduction.neighbors <- function(i,S,Neib){
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

# Create all segments
cReduction.offline.create.all.segments <- function(data, t) {
	print('Step 1: Create all segments')

	allSegs <- list()
	k <- 0
	cols <- ncol(data)
	rows <- length(unique(data[,1]))
	threshold <- (t * rows)

	for(i in 5:cols){
		data.ids <- as.vector(unique(data[,i]))
		for(j in data.ids){
			totest <- which(data[,i]==j)

			sessions <- as.vector(unique(data[totest,1]))
#			sessions <- NULL
#			for(kk in totest){
#				if((data[kk,1] %in% sessions)==FALSE){
#					sessions <- c(sessions, data[kk,1])
#				}
#			}

			if(length(sessions) >= threshold){
				k <- k + 1
				allSegs[[k]] <- list(dim=names(data[i]),val=j,idx=totest)	
			}
		}
	}
	allSegs
}

# Compute the contextual segments
cReduction.offline.create.contextual.segments <- function(data.train, seg, n, neigh){
	allSegsC <- list(list(),list(),list(),list(),list())

	# Step 2
	print('Step 2: Create contextual segments')
	num.seg <- length(seg)
	if(num.seg > 0){
		allSegsCtmp <- list()
		allSegsCtmpF1Values <- list()
		k <- 1

		for(i in 1:num.seg){
			# Segment - S
			name.seg <- paste(as.character(seg[[i]][[1]]), as.character(seg[[i]][[2]]), sep='_')
			data.seg <- data.train[seg[[i]][[3]],c(1,2,3,4)]
			dimensions.names <<- c(name.seg,'ibcf')

			folders <- c(1,2,3,4,5)
			result.all.folders <- NULL

			for(f in folders){
				data2validation <- cReduction.offline.data.setup(data.seg, f)

				result <- NULL
				result <- cReduction.offline.cf.exp(Train=unique(data2validation$train[,c(1,4)]), Test=unique(data2validation$test[,c(1,4)]), Test.ids=data2validation$test.ids, Hidden=data2validation$hidden, Table=name.seg, Field=name.seg, Ntop=n, Neib=neigh)

				data.train.ids <- unique(data.train[,1])
				new.data.train.ids <- setdiff(data.train.ids, data2validation$test.ids.all)
			
				# Total - T
				tses <- NULL
				turl <- NULL
				for(s in new.data.train.ids) {
					allUrl <- as.character(data.train[data.train[,1]==s,4])
					turl <- c(turl,allUrl)
					tses <- c(tses,rep(s,length(allUrl)))
				}
				data.t <- data.frame(tses,turl)
	
				result_tmp <- NULL
				result_tmp <- cReduction.offline.cf.exp(Train=unique(data.t[,c(1,2)]), Test=unique(data2validation$test[,c(1,4)]), Test.ids=data2validation$test.ids, Hidden=data2validation$hidden, Table='ibcf', Field='ibcf', Ntop=n, Neib=neigh)

				result.all.folders <- rbind(result.all.folders, result, result_tmp)
			}		

			# Test whether it is statistically significant
			ddim <- as.vector(unique(result.all.folders[,1]))
			ttop <- as.vector(unique(result.all.folders[,3]))

			for(ii in ddim) {      
				for(j in ttop) { 

					if(ii != 'ibcf'){
						odt <- result.all.folders[result.all.folders[,1] == ii & result.all.folders[,3] == j,]
						odtOnly_items <- result.all.folders[result.all.folders[,1] == 'ibcf' & result.all.folders[,3] == j,]

						pvalue <- t.test(as.numeric(odt[,9]), as.numeric(odtOnly_items[,9]), paired=T, alternative='greater', conf=0.95)[[3]]
						if( (!is.na(pvalue)) & (pvalue < 0.05) ){

							# Re-build the model with the entire segment
							ModelSs <- cReduction.simmatrix(unique(data.seg[,c(1,4)]))
							allSegsCtmp[[k]] <- list(mes=mean(as.numeric(odt[,9])), dim=seg[[i]][[1]], val=seg[[i]][[2]], its=as.vector(unique(data.seg[,4])), mdl=ModelSs, top=as.numeric(j))
							allSegsCtmpF1Values[[k]] <- list(mes=odt[,9])
							k <- k + 1

						}
					}
				}	
			}
		}

		# Step 3
		print('Step 3: Remove sub contextual segments')
		nseg <- length(allSegsCtmp)
		if(nseg > 0){
			kkk <- 1
			for(ss in n){
				allSegsCtmp.by.top <- list()
				allSegsCtmpF1Values.by.top <- list()
				kk <- 1
				for(i in 1:nseg){
					if(allSegsCtmp[[i]][[6]]==ss){
						allSegsCtmp.by.top[[kk]] <- allSegsCtmp[[i]]
						allSegsCtmpF1Values.by.top[[kk]] <- allSegsCtmpF1Values[[i]]
						kk <- kk + 1
					}
				}

				nseg.by.top <- length(allSegsCtmp.by.top)
				if(nseg.by.top > 1){
					allSegsCtmp1.by.top <- list()
					k4 <- 1
					for(i in 1:nseg.by.top){
						my.Previous <- allSegsCtmp.by.top[[i]][[4]]
						my.Previous.F1 <- allSegsCtmpF1Values.by.top[[i]][[1]]

						j <- 1
						stop <- FALSE
						while((j <= nseg.by.top) & (stop == FALSE)){
							if(i!=j){
								my.Next <- allSegsCtmp.by.top[[j]][[4]]
								my.Next.F1 <- allSegsCtmpF1Values.by.top[[j]][[1]]

								pvalue <- t.test(as.numeric(my.Next.F1), as.numeric(my.Previous.F1), paired=T, alternative='greater', conf=0.95)[[3]]
								if( (all(my.Previous %in% my.Next)==TRUE) & (!is.na(pvalue)) & (pvalue < 0.05) ){
									stop <- TRUE
								}
							}
							j <- j + 1
						}
						if(stop == FALSE){
							allSegsCtmp1.by.top[[k4]] <- allSegsCtmp.by.top[[i]]
							k4 <- k4 + 1
						}
					}

					# Sorting by F1 metric
					print('Step 3: Sort the contextual segments by F1')
					if(length(allSegsCtmp1.by.top) > 1){
						allF1 <- as.numeric(lapply(allSegsCtmp1.by.top, function(x){x[[1]]}))
						pos <- c(1:length(allSegsCtmp1.by.top))
						unordered <- data.frame(allF1,pos)
						ordered <- unordered[order(unordered[,1],decreasing = TRUE),]

						allSegsC.by.top <- list()
						for(i in 1:nrow(ordered)){
							allSegsC.by.top[[i]] <- allSegsCtmp1.by.top[[ordered[i,2]]]
						}
						allSegsC[[kkk]] <- allSegsC.by.top
						kkk <- kkk + 1
					} else {
						allSegsC[[kkk]] <- allSegsCtmp1.by.top
						kkk <- kkk + 1
					}
				} else {
					allSegsC[[kkk]] <- allSegsCtmp.by.top
					kkk <- kkk + 1
				}
			}
		} 
	}
	allSegsC
}

# Prepare data for the experiments - offline step
cReduction.offline.load.data.and.setup <- function(Dataset, Fold) {
	Dataset[,1] <- as.numeric(Dataset[,1])
	for(i in 4:ncol(Dataset)){
		Dataset[,i] <- as.character(Dataset[,i])
	}
	cReduction.offline.data.setup(Dataset,Fold)
}

# Prepare data for the experiments and create the folds - offline step
cReduction.offline.data.setup <- function(data, fold) {
	OUT<-NULL
	
	data <- data[order(data[,1]),]
	data.ids <- as.vector(unique(data[,1]))
	
	max <- length(data.ids)
	set.seed(20)
	sample <- rep(1:5, length=max)[order(runif(max))]

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
	
	OUT$test.ids.all <- test.ids
	test.ids.ok <- setdiff(test.ids,data.test.unique)
	data.hidd <- data.frame(sesId=data.hidd.idx, urlId=data.hidd.item)
	
	OUT$train <- data.train
	OUT$test <- data.test.all
	OUT$test.ids <- test.ids.ok
	OUT$hidden <- data.hidd
	OUT
}

# Prepare data for the experiments
cReduction.load.data.and.setup <- function(Dataset, Fold) {
	Dataset[,1] <- as.numeric(Dataset[,1])
	for(i in 4:ncol(Dataset)){
		Dataset[,i] <- as.character(Dataset[,i])
	}
	cReduction.data.setup(Dataset, Fold)
}

# Prepare data for the experiments and create the folds
cReduction.data.setup <- function(data, fold) {
	OUT <- NULL
	
	data <- data[order(data[,1]),]
	data.ids <- as.vector(unique(data[,1]))
	
	max <- length(data.ids)
	set.seed(20)
	sample <- rep(1:10, length=max)[order(runif(max))]

	train.ids <- data.ids[sample != fold]
	data.train <- NULL
	for(i in train.ids) { 
		data.partition <- data[which(data[,1]==i),]

		for(j in 5:ncol(data.partition)){
			item.dim <- unique(data.partition[,c(4,j)])
			tb <- as.data.frame(table(item.dim[,2])) # dim.val - freq
			dim.val <- as.vector(tb[which.max(tb[,2]),1])
			data.partition[,j] <- dim.val[1]
		}
		data.train <- rbind(data.train,data.partition)
	}
		
	test.ids <- data.ids[sample == fold]
	data.test.all <- list()
	data.test.unique <- NULL
	data.hidd.idx <- NULL
	data.hidd.item <- NULL
	k <- 1
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

			dimensions <- NULL
			for(j in 5:ncol(data.partition)){
				item.dim <- unique(data.partition[,c(4,j)])
				tb <- as.data.frame(table(item.dim[,2])) # dim.val - freq
				dim.val <- as.vector(tb[which.max(tb[,2]),1])
				dimensions <- c(dimensions,paste(names(data.partition[j]),dim.val[1],sep='='))
			}

			data.test.all[[k]] <- list(ses=i,its=as.vector(unique(data.partition[,4])),dim=dimensions)
			k <- k + 1
		} else {
			data.test.unique <- c(data.test.unique,i)	
		}
	}
	
	test.ids.ok <- setdiff(test.ids,data.test.unique)
	data.hidd <- data.frame(sesId=data.hidd.idx, urlId=data.hidd.item)
	
	OUT$train <- data.train
	OUT$test <- data.test.all
	OUT$test.ids <- test.ids.ok
	OUT$hidden <- data.hidd

	OUT
}

# Run IBCF algorithm for the offline step
cReduction.offline.cf.exp <- function(Train=NULL, Test=NULL, Test.ids=NULL, Hidden=NULL, Table='', Field='', Ntop=c(1), Neib=2){
	results <- NULL
	S <- cReduction.simmatrix(Train)
	results <- cReduction.offline.cf.reeval(S,Test,Test.ids,Hidden,Table,Field,Ntop,Neib)
	results
}

# Take a test set, a hidden set and a similarity matrix
# and outputs evaluation statistics for each topN - offline step
cReduction.offline.cf.reeval <- function(S,Test,Test.Ids,Hid,Table,Field,Ntop,Neib) {
	results <- NULL
	Recs <- cReduction.offline.topNrec.batch(Test,Test.Ids,S,Ntop,Neib) # Recommendations for each topN
	
	uids <- NULL
	recs <- NULL
	for(i in 1:length(Recs)){
		idx <- Recs[[i]][[1]]
		items <- (Recs[[i]][[2]])[[1]]
		recs <- c(recs,items)
		uids <- c(uids,rep(idx,length(items)))		
	}
	RecsTmp <- data.frame(sesId=uids,urlId=recs)
	res <- cReduction.rec.eval(RecsTmp,Hid,Test.Ids,S,Test) # Evaluate the recommendations and return statistics
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
		res <- cReduction.rec.eval(RecsTmp,Hid,Test.Ids,S,Test) # Evaluate the recommendations and return statistics
		results <- rbind(results, c(Table,Field,Ntop[j],res$hits,res$nrecs,res$nhidden,res$recall,res$precision,res$f1,res$fallout,res$map))
	}
	results
}

# Gives topN recommendations for each user session - offline step
cReduction.offline.topNrec.batch <- function(Obs,ObsIds,S,N,Neib) {
	Usr <- as.vector(ObsIds)

	allTopNrec <- list()
	idxtopNrec <- 1
	
	for(u in Usr) {
		lrec <- cReduction.offline.topNrec(Obs[Obs[,1]==u,2],S,N,Neib)
		oneTopNrec <- list(idx=u,recs=lrec)
		allTopNrec[[idxtopNrec]] <- oneTopNrec
		idxtopNrec <- idxtopNrec + 1
	}
	allTopNrec
}

# Gives topN recommendations for one user session - offline step
cReduction.offline.topNrec <- function(R,S,N,Neib) {
	recs <- cReduction.offline.wsum(R,S,Neib)

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

# Retrieve the score for each candidate recommendation for a user session - offline step
cReduction.offline.wsum <- function(R,S,Neib) {
	idx <- names(S)
	sdf <- setdiff(idx,R)
	n <- length(sdf)

	B <- sapply(1:n, function(i){ 
		cReduction.offline.wsumitem(R,S,sdf[i],Neib)
	})
	names(B) <- sdf
	B
}

# Compute the score for one candidate recommendation - offline step
cReduction.offline.wsumitem <- function(R,S,i,neighb) {
	Neighbs <- cReduction.offline.neighbors(i,S,neighb)
	
	N <- intersect(names(Neighbs),R)
	num <- sum(Neighbs[N])
	den <- sum(Neighbs)

	if (den==0) 0
	else num / den
}

# Find the neighbors - offline step
cReduction.offline.neighbors <- function(i,S,Neib){
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
cReduction.rec.eval <- function(Recs, Hid, ObsIds, S, Obs){
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
		hits_tmp <- cReduction.hits.per.user(Recs,Hid,u)
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
cReduction.hits.per.user <- function(R,H,u) {
	length(intersect(R[R[,1]==u,2],H[H[,1]==u,2]))
}

# Compute the similarity matrix using adjacency list
cReduction.simmatrix <- function(A) {
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


################### FUNCOES DE ANALISE ############################
write.segmentsC <- function(data,segC,N){
	if(length(segC) > 0){
		for(i in 1:length(segC)){
			data.seg.C <- data[data[,segC[[i]][[2]]]==segC[[i]][[3]], c(1,2,3,4)]
			write.table(data.seg.C, file = paste(segment.name,'/top',N,'_',segC[[i]][[2]],'_',segC[[i]][[3]],'.bak',sep=''), row.names = FALSE, col.names = T, quote = FALSE, sep = " ")
		}
	}
}
###################################################################

# run all folders in parallel
cReduction.run.all.folders.parallel()
# run all folders sequentially
# cReduction.run.all.folders()


# Independent ML/Laplace mixed-model baseline on Python-prepared designs.
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(jsonlite))
tr <- read.csv(args[1]); te <- read.csv(args[2])
tr$g <- factor(tr$g); te$g <- factor(te$g)
f <- as.formula(args[3]); task <- args[4]; regime <- args[5]
warn <- character()
fit <- withCallingHandlers({
  if (task == 'regression') {
    lmer(f, data=tr, REML=FALSE,
         control=lmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000)))
  } else {
    glmer(f, data=tr, family=binomial(), nAGQ=1,
          control=glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000)))
  }
}, warning=function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart('muffleWarning') })
# Known groups use conditional coefficients; unseen groups use zero RE.
pred <- predict(fit, newdata=te, type='response',
                re.form=if(regime=='A') NULL else NA, allow.new.levels=TRUE)
messages <- fit@optinfo$conv$lme4$messages
opt_code <- fit@optinfo$conv$opt
# A boundary/singular fit is reported separately from optimizer failure.
nonboundary <- messages[!grepl('singular', messages)]
write_json(list(prediction=unname(pred), singular=isSingular(fit),
                converged=all(opt_code==0) && length(nonboundary)==0,
                messages=unique(c(warn,messages)),
                lme4_version=as.character(packageVersion('lme4')),
                R_version=R.version.string), args[6], auto_unbox=TRUE, digits=16)

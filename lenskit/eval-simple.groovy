import org.grouplens.lenskit.eval.data.crossfold.RandomOrder

import org.grouplens.lenskit.knn.NeighborhoodSize
import org.grouplens.lenskit.knn.item.*
import org.grouplens.lenskit.knn.user.*
import org.grouplens.lenskit.vectors.similarity.*
import org.grouplens.lenskit.mf.funksvd.*
import org.grouplens.lenskit.mf.*
import org.grouplens.lenskit.iterative.*

import org.apache.commons.lang3.BooleanUtils
import org.grouplens.lenskit.transform.normalize.*
import org.grouplens.lenskit.baseline.*

def zipFile = 'ml100k.zip'
def dataDir = 'ml100k'

// This target unpacks the data
target('download') {
    perform {
        // check if the data license is acknowledged
        if (!BooleanUtils.toBoolean(config["grouplens.mldata.acknowledge"])) {
            logger.error(
                    "This analysis makes use of the MovieLens 100K data " +
                            "set from GroupLens Research. Use of this data set is restricted to " +
                            "non-commercial purposes and is only permitted in accordance with the " +
                            "license terms. To use this data in LensKit's automated tests, set the " +
                            "`grouplens.mldata.acknowledge' property to `yes' to indicate you " +
                            "acknowledge the usage license.  More information is available at " +
                            "<http://www.grouplens.org/node/73>.")
            throw new RuntimeException("GroupLens data license not acknoweldged");
        }
    }
    ant.get(src: 'http://www.grouplens.org/system/files/ml-100k.zip',
            dest: zipFile,
            skipExisting: true)
    ant.unzip(src: zipFile, dest: dataDir) {
        patternset {
            include name: 'ml-100k/*'
        }
        mapper type: 'flatten'
    }
}

// this target cross-folds the data. The target object can be used as the data set; it holds
// the value of the last task (in this case, 'crossfold').  The crossfold won't actually be
// avaiable until it is executed, but the evaluator automatically takes care of that.
def ml100k = target('crossfold') {
    requires 'download'
    crossfold {
        source csvfile("${dataDir}/u.data") {
            delimiter "\t"
            domain {
                minimum 1.0
                maximum 5.0
                precision 1.0
            }
        }
        train "ml100k-crossfold/train.%d.csv"
        test "ml100k-crossfold/test.%d.csv"
        order RandomOrder
        holdout 10
        partitions 5
    }
}

target('evaluate') {
    // this requires the ml100k target to be run first
    // can either reference a target by object or by name (as above)
    requires ml100k

    trainTest {
        dataset ml100k

        // Three different types of output for analysis.
        output "eval-results.csv"
        predictOutput "eval-preds.csv"
        userOutput "eval-user.csv"

        metric CoveragePredictMetric
        metric RMSEPredictMetric
        metric NDCGPredictMetric
	metric MAEPredictMetric

        algorithm("ItemItemCosine") {
            // use the item-item rating predictor with a baseline and normalizer
            bind ItemScorer to ItemItemScorer
	    within(ItemSimilarity){
		bind VectorSimilarity to CosineVectorSimilarity
	   }
        }

	Algorithm("ItemItemPearson") {
              // use the item-item rating predictor with a baseline and normalizer
              bind ItemScorer to ItemItemScorer
              within(ItemSimilarity){
                  bind VectorSimilarity to PearsonCorrelation
             }
          }


        algorithm("UserUserCosineN10") {
            bind ItemScorer to UserUserItemScorer
            set NeighborhoodSize to 10
	    bind NeighborhoodFinder to SimpleNeighborhoodFinder
	    within(ItemSimilarity){
		bind VectorSimilarity to CosineVectorSimilarity
	    }
        }


        algorithm("UserUserCosineN50") {
            bind ItemScorer to UserUserItemScorer
            set NeighborhoodSize to 50
	    bind NeighborhoodFinder to SimpleNeighborhoodFinder
	    within(ItemSimilarity){
		bind VectorSimilarity to CosineVectorSimilarity
	    }
        }

        algorithm("UserUserPearsonN10") {
            bind ItemScorer to UserUserItemScorer
            set NeighborhoodSize to 10
	    bind NeighborhoodFinder to SimpleNeighborhoodFinder
	    within(ItemSimilarity){
		bind VectorSimilarity to PearsonCorrelation
	    }
        }

        algorithm("UserUserPearsonN50") {
            bind ItemScorer to UserUserItemScorer
            set NeighborhoodSize to 50
	    bind NeighborhoodFinder to SimpleNeighborhoodFinder
	    within(ItemSimilarity){
		bind VectorSimilarity to PearsonCorrelation
	    }
        }


	algorithm("SVDFunk10factors"){
	  bind ItemScorer to FunkSVDItemScorer
	  bind (BaselineScorer, ItemScorer) to UserMeanItemScorer
	  bind StoppingCondition to IterationCountStoppingCondition
	  set IterationCount to 50
	  set FeatureCount to 10
	}

	algorithm("SVDFunk50factors"){
	  bind ItemScorer to FunkSVDItemScorer
	  bind (BaselineScorer, ItemScorer) to UserMeanItemScorer
	  bind StoppingCondition to IterationCountStoppingCondition
	  set IterationCount to 50
	  set FeatureCount to 10
	}


    }
}

// After running the evaluation, let's analyze the results
target('analyze') {
    requires 'evaluate'
    // Run R. Note that the script is run in the analysis directory; you might want to
    // copy all R scripts there instead of running them from the source dir.
    ant.exec(executable: config["rscript.executable"]) {
        arg value: "chart.R"
    }
}

// By default, run the analyze target
defaultTarget 'analyze'

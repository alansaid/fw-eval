/**
 * Created with IntelliJ IDEA.
 * User: alan
 * Date: 2014-01-15
 * Time: 18:02
 */

package net.recommenders.rival.extras;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDPlusPlusFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.UncenteredCosineSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDPlusPlusFactorizer;
import java.io.File;

public class MahoutEvaluator {


    public static void main(String[] args){

        String file = args[0];
        System.out.println(file);
        int at = 10;
        MahoutEvaluator me = new MahoutEvaluator();
        try {
        //    me.runUserKnn(file, at);
         //   me.runItemBased(file, at);
            me.runSVD(file, at);
        } catch (Exception e){
            e.printStackTrace();
        }

    }

    public void runUserKnn(String fileName, int at) throws Exception{
        DataModel dm = new FileDataModel(new File(fileName));

        RandomUtils.useTestSeed();
        RecommenderIRStatsEvaluator evaluator =
                new GenericRecommenderIRStatsEvaluator ();

        /** PEARSON **/
        RecommenderBuilder recommenderBuilderN10 = new RecommenderBuilder() {
            int neighborhoodSize = 10;
            @Override
            public Recommender buildRecommender(DataModel model)throws TasteException {
                UserSimilarity similarity = new PearsonCorrelationSimilarity (model);
                UserNeighborhood neighborhood = new NearestNUserNeighborhood(neighborhoodSize, similarity, model);
                return new GenericUserBasedRecommender(model, neighborhood, similarity);
            }
        };
        IRStatistics stats10 = evaluator.evaluate(
                recommenderBuilderN10, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        String output = "Userbased Neighborhood size = 10, pearson \n" +
                "P@" + at + ": " + stats10.getPrecision() + "\n" +
                "R@" + at + ": " + stats10.getRecall() + "\n" +
                "ndcg: " + stats10.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats10.getReach() + "\n";

        at = 50;
        stats10 = evaluator.evaluate(
                recommenderBuilderN10, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Userbased Neighborhood size = 10, pearson \n" +
                "P@" + at + ": " + stats10.getPrecision() + "\n" +
                "R@" + at + ": " + stats10.getRecall() + "\n" +
                "ndcg: " + stats10.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats10.getReach() + "\n";

        RecommenderBuilder recommenderBuilderN50 = new RecommenderBuilder() {
            int neighborhoodSize = 50;
            @Override
            public Recommender buildRecommender(DataModel model)throws TasteException {
                UserSimilarity similarity = new PearsonCorrelationSimilarity (model);
                UserNeighborhood neighborhood = new NearestNUserNeighborhood(neighborhoodSize, similarity, model);
                return new GenericUserBasedRecommender(model, neighborhood, similarity);
            }
        };
        at = 10;
        IRStatistics stats50 = evaluator.evaluate(
                recommenderBuilderN50, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Userbased Neighborhood size = 50, pearson\n" +
                "P@" + at + ": " + stats50.getPrecision() + "\n" +
                "R@" + at + ": " + stats50.getRecall() + "\n" +
                "ndcg: " + stats50.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats50.getReach() + "\n";
        at = 50;
        stats50 = evaluator.evaluate(
                recommenderBuilderN50, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Userbased Neighborhood size = 50, pearson\n" +
                "P@" + at + ": " + stats50.getPrecision() + "\n" +
                "R@" + at + ": " + stats50.getRecall() + "\n" +
                "ndcg: " + stats50.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats50.getReach() + "\n";


        /** COSINE **/
        RecommenderBuilder recommenderBuilderN10C = new RecommenderBuilder() {
            int neighborhoodSize = 10;
            @Override
            public Recommender buildRecommender(DataModel model)throws TasteException {
                UserSimilarity similarity = new UncenteredCosineSimilarity(model);
                UserNeighborhood neighborhood = new NearestNUserNeighborhood(neighborhoodSize, similarity, model);
                return new GenericUserBasedRecommender(model, neighborhood, similarity);
            }
        };
        at = 10;
        IRStatistics stats10C = evaluator.evaluate(
                recommenderBuilderN10C, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Userbased Neighborhood size = 10, cosine \n" +
                "P@" + at + ": " + stats10.getPrecision() + "\n" +
                "R@" + at + ": " + stats10.getRecall() + "\n" +
                "ndcg: " + stats10.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats10.getReach() + "\n";
        at = 50;
        stats10C = evaluator.evaluate(
                recommenderBuilderN10C, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Userbased Neighborhood size = 10, cosine \n" +
                "P@" + at + ": " + stats10.getPrecision() + "\n" +
                "R@" + at + ": " + stats10.getRecall() + "\n" +
                "ndcg: " + stats10.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats10.getReach() + "\n";

        RecommenderBuilder recommenderBuilderN50C = new RecommenderBuilder() {
            int neighborhoodSize = 50;
            @Override
            public Recommender buildRecommender(DataModel model)throws TasteException {
                UserSimilarity similarity = new UncenteredCosineSimilarity (model);
                UserNeighborhood neighborhood = new NearestNUserNeighborhood(neighborhoodSize, similarity, model);
                return new GenericUserBasedRecommender(model, neighborhood, similarity);
            }
        };
        at = 10;
        IRStatistics stats50C = evaluator.evaluate(
                recommenderBuilderN50C, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Userbased Neighborhood size = 50, cosine\n" +
                "P@" + at + ": " + stats50.getPrecision() + "\n" +
                "R@" + at + ": " + stats50.getRecall() + "\n" +
                "ndcg: " + stats50.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats50.getReach() + "\n";
        at = 50;
        stats50C = evaluator.evaluate(
                recommenderBuilderN50C, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Userbased Neighborhood size = 50, cosine\n" +
                "P@" + at + ": " + stats50.getPrecision() + "\n" +
                "R@" + at + ": " + stats50.getRecall() + "\n" +
                "ndcg: " + stats50.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats50.getReach() + "\n";
        System.out.println(output);
    }

    public void runItemBased(String fileName, int at) throws Exception{
        DataModel dm = new FileDataModel(new File(fileName));

        RandomUtils.useTestSeed();
        RecommenderIRStatsEvaluator evaluator =
                new GenericRecommenderIRStatsEvaluator ();

        /** PEARSON **/
        RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
            @Override
            public Recommender buildRecommender(DataModel model)throws TasteException {
                ItemSimilarity similarity = new PearsonCorrelationSimilarity(model);
                return new GenericItemBasedRecommender(model, similarity);
            }
        };
        IRStatistics stats = evaluator.evaluate(
                recommenderBuilder, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        String output = "Itembased, pearson \n" +
                "P@" + at + ": " + stats.getPrecision() + "\n" +
                "R@" + at + ": " + stats.getRecall() + "\n" +
                "ndcg: " + stats.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats.getReach() + "\n";
        at = 50;
        stats = evaluator.evaluate(
                recommenderBuilder, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Itembased, pearson \n" +
                "P@" + at + ": " + stats.getPrecision() + "\n" +
                "R@" + at + ": " + stats.getRecall() + "\n" +
                "ndcg: " + stats.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats.getReach() + "\n";

        /** COSINE **/
        RecommenderBuilder recommenderBuilderC = new RecommenderBuilder() {
            @Override
            public Recommender buildRecommender(DataModel model)throws TasteException {
                ItemSimilarity similarity = new UncenteredCosineSimilarity(model);
                return new GenericItemBasedRecommender(model, similarity);
            }
        };
        at = 10;
        IRStatistics statsC = evaluator.evaluate(
                recommenderBuilderC, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Itembased, cosine \n" +
                "P@" + at + ": " + statsC.getPrecision() + "\n" +
                "R@" + at + ": " + statsC.getRecall() + "\n" +
                "ndcg: " + statsC.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + statsC.getReach() + "\n";
        at = 50;
        statsC = evaluator.evaluate(
                recommenderBuilderC, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "Itembased, cosine \n" +
                "P@" + at + ": " + statsC.getPrecision() + "\n" +
                "R@" + at + ": " + statsC.getRecall() + "\n" +
                "ndcg: " + statsC.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + statsC.getReach() + "\n";


        System.out.println(output);
    }

    public void runSVD(String fileName, int at) throws Exception{
        DataModel dm = new FileDataModel(new File(fileName));

        RandomUtils.useTestSeed();
        RecommenderIRStatsEvaluator evaluator =
                new GenericRecommenderIRStatsEvaluator ();


        RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
            @Override
            public Recommender buildRecommender(DataModel model)throws TasteException {
                Factorizer fact = new SVDPlusPlusFactorizer(model,10, 50);
                return new SVDRecommender(model, fact);
            }
        };
        IRStatistics stats = evaluator.evaluate(
                recommenderBuilder, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        String output = "SVD, 10 factors \n" +
                "P@" + at + ": " + stats.getPrecision() + "\n" +
                "R@" + at + ": " + stats.getRecall() + "\n" +
                "ndcg: " + stats.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats.getReach() + "\n";
        at = 50;
        stats = evaluator.evaluate(
                recommenderBuilder, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "SVD, 10 factors \n" +
                "P@" + at + ": " + stats.getPrecision() + "\n" +
                "R@" + at + ": " + stats.getRecall() + "\n" +
                "ndcg: " + stats.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats.getReach() + "\n";


        RecommenderBuilder recommenderBuilder50 = new RecommenderBuilder() {
            @Override
            public Recommender buildRecommender(DataModel model)throws TasteException {
                Factorizer fact = new SVDPlusPlusFactorizer(model,50, 50);
                return new SVDRecommender(model, fact);
            }
        };
        at = 10;
        IRStatistics stats50 = evaluator.evaluate(
                recommenderBuilder50, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "SVD, 50 factors \n" +
                "P@" + at + ": " + stats50.getPrecision() + "\n" +
                "R@" + at + ": " + stats50.getRecall() + "\n" +
                "ndcg: " + stats50.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats50.getReach() + "\n";
        at = 50;
        stats = evaluator.evaluate(
                recommenderBuilder, null, dm, null, at,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
                1.0);
        output += "SVD, 50 factors \n" +
                "P@" + at + ": " + stats50.getPrecision() + "\n" +
                "R@" + at + ": " + stats50.getRecall() + "\n" +
                "ndcg: " + stats50.getNormalizedDiscountedCumulativeGain() + "\n" +
                "Reach: " + stats50.getReach() + "\n";





        System.out.println(output);
    }


}

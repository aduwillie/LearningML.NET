using Microsoft.ML;
using Microsoft.ML.Data;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace BinarayClassification.SentimentAnalysis;

internal sealed class Trainer
{
    private readonly MLContext mlContext;
    private TrainTestData? splitDataView;
    private ITransformer? model;

    public const string FeatureColumnName = "Features";
    public const string LabelColumnName = "Label";
    public const string ProbabilityColumnName = "Probability";
    public const string PredictedLabelColumnName = "PredictedLabel";
    public const string ScoreColumnName = "Score";

    public Trainer()
    {
        mlContext = new MLContext();
    }

    public ITransformer Train(string dataFilePath, double testFraction = 0.2)
    {
        var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(dataFilePath);
        splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction);

        var estimator = mlContext.Transforms
            .Text
            // Convert sentiment text column to numerical features eg. [0.76, 0.64, 0.28,...]
            .FeaturizeText(
                outputColumnName: FeatureColumnName,
                inputColumnName: nameof(SentimentData.SentimentText));

        var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: LabelColumnName,
            featureColumnName: FeatureColumnName);

        var pipeline = estimator.Append(trainer);

        model = pipeline.Fit(splitDataView?.TrainSet);
        return model;
    }

    public string Evaluate()
    {
        if (model is null)
        {
            throw new Exception("Cannot perform evaluation before training a model.");
        }

        var predictions = model.Transform(splitDataView?.TestSet);
        CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, LabelColumnName);

        var output = new StringBuilder("\n");
        output.Append("Model quality metrics evaluation\n");
        output.Append("--------------------------------\n");
        output.Append($"Accuracy: {metrics.Accuracy:P2}\n");
        output.Append($"Auc: {metrics.AreaUnderRocCurve:P2}\n");
        output.Append($"F1Score: {metrics.F1Score:P2}\n");
        output.Append("=============== End of model evaluation ===============");

        return output.ToString();
    }

    public string Predict(string input)
    {
        var inputData = new SentimentData { SentimentText = input };
        var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        var result = predictionEngine.Predict(inputData);
        var output = new StringBuilder("\n");
        output.Append($"Prediction: {input}\n");
        output.Append($"Sentiment: {result.SentimentText}\n");
        output.Append($"Prediction: {(Convert.ToBoolean(result.Prediction) ? "Positive" : "Negative")}\n");
        output.Append($"Probability: {result.Probability:P2}\n");

        return output.ToString();
    }
}

using Microsoft.ML;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace MultiClassification.IssueCategorizer;

internal class Trainer
{
    private readonly MLContext mlContext;
    private ITransformer? model;

    private TrainTestData? splitDataView;

    private const string AreaColumnName = "Area";
    private const string TitleFeaturizedColumnName = "TitleFeaturized";
    private const string TitleColumnName = "Title";
    private const string DescriptionColumnName = "Description";
    private const string DescriptionFeaturizedColumnName = "DescriptionFeaturized";
    private const string LabelColumnName = "Label";
    private const string FeaturesColumnName = "Features";
    private const string PredictedLabelColumnName = "PredictedLabel";

    public Trainer()
    {
        mlContext = new MLContext();
    }

    public ITransformer Train(string dataFilePath, double testFraction = 0.2)
    {
        var dataView = mlContext.Data.LoadFromTextFile<GithubIssue>(dataFilePath);
        splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction);

        var pipeline = mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: LabelColumnName, inputColumnName: AreaColumnName)
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: TitleFeaturizedColumnName, inputColumnName: TitleColumnName))
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: DescriptionFeaturizedColumnName, inputColumnName: DescriptionColumnName))
            .Append(mlContext.Transforms.Concatenate(FeaturesColumnName, TitleFeaturizedColumnName, DescriptionFeaturizedColumnName))
            // ONLY needed for smaller datasets
            .AppendCacheCheckpoint(mlContext)
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
            labelColumnName: LabelColumnName,
            featureColumnName: FeaturesColumnName)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelColumnName)));

        model = pipeline.Fit(splitDataView.Value.TrainSet);
        return model;
    }

    public string Evaluate(string? dataFilePath)
    {
        if (model is null)
        {
            throw new Exception("Cannot perform evaluation before training a model.");
        }

        IDataView? testData = splitDataView?.TestSet;
        if (dataFilePath is not null)
        {
            var data = mlContext.Data.LoadFromTextFile<GithubIssue>(dataFilePath);
            testData = mlContext.Data.TrainTestSplit(data, 1).TrainSet;
        }

        var predictions = model.Transform(testData);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions, LabelColumnName);

        var output = new StringBuilder("\n");
        output.Append("Model quality metrics evaluation\n");
        output.Append("--------------------------------\n");
        output.Append($"Accuracy: {metrics.MicroAccuracy:P2}\n");
        output.Append($"Accuracy: {metrics.MacroAccuracy:P2}\n");
        output.Append($"Auc: {metrics.LogLoss:#.###}\n");
        output.Append($"F1Score: {metrics.LogLossReduction:#.###}\n");
        output.Append("=============== End of model evaluation ===============");

        return output.ToString();
    }

    public string Predict(string title, string description)
    {
        var issueToPredict = new GithubIssue
        {
            Title = title,
            Description = description
        };

        var predictionEngine = mlContext.Model.CreatePredictionEngine<GithubIssue, IssuePrediction>(model);
        var result = predictionEngine.Predict(issueToPredict);

        var output = new StringBuilder("\n");
        output.Append($"Prediction: Title: {title}, Description: {description}\n");
        output.Append($"Sentiment: {result.Area}\n");
        output.Append($"Probability: {result.Probability:P2}\n");
        output.Append($"Probability: {result.Score}\n");

        return output.ToString();
    }
}

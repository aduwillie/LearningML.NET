using Microsoft.ML.Data;

namespace BinarayClassification.SentimentAnalysis;

internal class SentimentData
{
    [LoadColumn(0)]
    public string? SentimentText { get; set; }
    [LoadColumn(1), ColumnName(Trainer.LabelColumnName)]
    public bool Sentiment { get; set; }
}

internal class SentimentPrediction : SentimentData
{
    [ColumnName(Trainer.PredictedLabelColumnName)]
    public bool Prediction { get; set; }
    public float Probability { get; set; }
    [ColumnName("Score")]
    public float Score { get; set; }
}

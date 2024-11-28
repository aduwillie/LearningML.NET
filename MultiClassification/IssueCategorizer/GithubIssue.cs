using Microsoft.ML.Data;

namespace MultiClassification.IssueCategorizer;

internal class GithubIssue
{
    [LoadColumn(0)]
    public string? ID { get; set; }
    [LoadColumn(1)]
    public string? Area { get; set; }
    [LoadColumn(2)]
    public required string Title { get; set; }
    [LoadColumn(3)]
    public required string Description { get; set; }
}

internal class IssuePrediction
{
    [ColumnName("PredictedLabel")]
    public string? Area { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}

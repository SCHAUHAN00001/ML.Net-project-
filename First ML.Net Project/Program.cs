using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace First_ML.Net_Project
{
    internal class Program
    {
        public class ComputerData
        {
            public float RAM { get; set; }
            public float SSD { get; set; }
            public float ICore { get; set; }
            public string Processor { get; set; }
            public float Price { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            ComputerData[] computerData =
            {
                new ComputerData() { RAM = 8, SSD = 256, ICore = 3.2F, Processor = "I3", Price = 45000 },
                new ComputerData() { RAM = 16, SSD = 512, ICore = 3.6F, Processor = "I5", Price = 55000 },
                new ComputerData() { RAM = 32, SSD = 1024, ICore = 4.0F, Processor = "I7", Price = 105000 },
                new ComputerData() { RAM = 64, SSD = 2048, ICore = 4.4F, Processor = "I9", Price = 125000 },
            };

            IDataView trainingData = mlContext.Data.LoadFromEnumerable(computerData);

            var pipeline = mlContext.Transforms.Concatenate("Features",
                new[] { "RAM", "SSD", "ICore" })
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            var model = pipeline.Fit(trainingData);

            ComputerData testData = new ComputerData()
            {
                RAM = 100,
                SSD = 4096,
                ICore = 5.0F,
                Processor = "I10"
            };

            Prediction price = mlContext.Model.CreatePredictionEngine<ComputerData, Prediction>(model).Predict(testData);
            Debug.WriteLine($"Predicted price for RAM: {testData.RAM}, SSD: {testData.SSD}, ICore: {testData.ICore}, Processor: {testData.Processor} = {price.Price:C}");
        }
    }
}






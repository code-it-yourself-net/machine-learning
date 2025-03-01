using System;

namespace MnistTests
{
    internal class Program
    {
        // consts
        public const int RandomSeed = 241030;
        public const int Epochs = 10;
        public const int BatchSize = 100;

        private static void Main(string[] args)
        {
            Console.WriteLine("Select a routine to run:");
            Console.WriteLine("1. ProgramMatrix");
            Console.WriteLine("2. ProgramTyped");
            Console.WriteLine("3. ProgramConv2D");
            // exit
            Console.WriteLine("0. Exit");
            Console.Write("Enter your choice: ");
            string? choice = Console.ReadLine();

            switch (choice)
            {
                case "1":
                    ProgramMatrix.Run();
                    break;
                case "2":
                    ProgramTyped.Run();
                    break;
                case "3":
                    ProgramConv2D.Run();
                    break;
                case "0":
                    Console.WriteLine("Goodbye!");
                    break;
                default:
                    Console.WriteLine("Invalid choice.");
                    break;
            }
        }
    }
}

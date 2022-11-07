using System;
using System.Collections.Generic;
using System.Threading;

using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Optimizer;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Utils
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class RoundArray< T >
    {
        private readonly T[] _Array;
        private int currentIdx = 0;
        public RoundArray( T[] a ) => _Array = a;
        public T GetNextItem()
        {
            var t = _Array[ currentIdx ];
            currentIdx = (currentIdx + 1) % _Array.Length;
            return (t);
        }
        public T[] ToArray() => _Array;
    }
    /// <summary>
    /// 
    /// </summary>
    public static class Misc
    {
        public static string GetTimeStamp( DateTime timeStamp ) => string.Format( "{0:yyyy}.{0:MM}.{0:dd}, {0:HH}.{0:mm}.{0:ss}", timeStamp );

        public static void StatusUpdateWatcher( object sender, CostEventArg e )
        {
            var now           = DateTime.Now;
            var total_elapsed = now - e.StartDateTime;
            var last_elapsed  = now - e.LastCallStatusUpdateWatcherDateTime;
            var sentPerMin = (0 < total_elapsed.TotalMinutes) ? (e.ProcessedSentencesInTotal / total_elapsed.TotalMinutes) : 0;
            var wordPerSec = (0 < total_elapsed.TotalSeconds) ? (e.ProcessedWordsInTotal     / total_elapsed.TotalSeconds) : 0;

            Logger.WriteLine( $"Update = {e.Update}, Epoch = {e.Epoch}, LR = {e.LearningRate:F6}, AvgCost = {e.AvgCostInTotal:F4}, Sent = {e.ProcessedSentencesInTotal}, SentPerMin = {sentPerMin:F}, WordPerSec = {wordPerSec:F}, ({last_elapsed}, total: {total_elapsed})" );
        }

        public static IOptimizer CreateOptimizer( Options opts )
        {
            // Create optimizer
            IOptimizer optimizer;
            if ( opts.OptimizerType == OptimizerTypeEnums.Adam )
            {
                optimizer = new AdamOptimizer( opts.GradClip, opts.Beta1, opts.Beta2 );
            }
            else
            {
                optimizer = new RMSPropOptimizer( opts.GradClip, opts.Beta1 );
            }
            return (optimizer);
        }

        [M(O.AggressiveInlining)] public static bool IsNullOrEmpty( this string s ) => string.IsNullOrEmpty( s );
        [M(O.AggressiveInlining)] public static bool IsNullOrWhiteSpace( this string s ) => string.IsNullOrWhiteSpace( s );
        [M(O.AggressiveInlining)] public static List< T > ToList< T >( this IEnumerable< T > seq, int capatity )
        {
            var lst = new List< T >( capatity );
            lst.AddRange( seq );
            return (lst);
        }
        [M(O.AggressiveInlining)] public static T[] ToArray< T >( this IEnumerable< T > seq, int count )
        {
            var array = new T[ count ];
            count = 0;
            foreach ( var t in seq )
            {
                array[ count++ ] = t;
            }
            return (array);
        }

        [M(O.AggressiveInlining)] public static bool AnyEx< T >( this T[] array ) => (array != null) && (0 < array.Length);
        [M(O.AggressiveInlining)] public static bool AnyEx< T >( this IList< T > lst ) => (lst != null) && (0 < lst.Count);
        [M(O.AggressiveInlining)] public static bool AnyEx< T >( this List< T > lst ) => (lst != null) && (0 < lst.Count);
        [M(O.AggressiveInlining)] public static bool AnyEx< T >( this ICollection< T > lst ) => (lst != null) && (0 < lst.Count);
        [M(O.AggressiveInlining)] public static bool AnyEx< T >( this IReadOnlyList< T > lst ) => (lst != null) && (0 < lst.Count);
        [M(O.AggressiveInlining)] public static bool AnyEx< T >( this IReadOnlyCollection< T > lst ) => (lst != null) && (0 < lst.Count);
        [M(O.AggressiveInlining)] public static bool AnyEx< K, T >( this IDictionary< K, T > lst ) => (lst != null) && (0 < lst.Count);
        [M(O.AggressiveInlining)] public static bool AnyEx< K, T >( this Dictionary< K, T > lst ) => (lst != null) && (0 < lst.Count);

        [M(O.AggressiveInlining)] public static void AddWithLock< K, T >( this IDictionary< K, T > d, K k, T t )
        {
            lock ( d )
            {
                d.Add( k, t );
            }
        }

        [M(O.AggressiveInlining)] public static void Console_Title( string title )
        {
            if ( OperatingSystem.IsWindows() )
            {
                Console.Title = title;
            }
        }
        [M(O.AggressiveInlining)] public static string Console_Title() => OperatingSystem.IsWindows() ? Console.Title : null;
    }

    /// <summary>
    /// 
    /// </summary>
    public struct Console_CancelKeyPress_Breaker : IDisposable
    {
        private CancellationTokenSource _Cts;
        private string _CancelMessage;
        private Console_CancelKeyPress_Breaker( CancellationTokenSource cts, string cancelMessage )
        {
            _Cts           = cts;
            _CancelMessage = cancelMessage;
            Console.CancelKeyPress += ConsoleCancelEventHandler;
        }
        public void Dispose()
        {
            Console.CancelKeyPress -= ConsoleCancelEventHandler;
            _Cts.Dispose();
        }
        private void ConsoleCancelEventHandler( object s, ConsoleCancelEventArgs e )
        {
            if ( _CancelMessage != null ) Console.WriteLine( _CancelMessage );
            e.Cancel = true;
            try { _Cts.Cancel(); } catch {; }
        }
        public CancellationTokenSource Cts => _Cts;
        public CancellationToken Token => _Cts.Token;
        public bool IsCancellationRequested => _Cts.IsCancellationRequested;

        public static Console_CancelKeyPress_Breaker Create( string cancelMessage = "\r\n cancel operation..." ) => new Console_CancelKeyPress_Breaker( new CancellationTokenSource(), cancelMessage );
    }
}

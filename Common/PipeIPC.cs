using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Threading;
using System.Threading.Tasks;

namespace Lingvo.PosTagger
{
    /// <summary>
    /// 
    /// </summary>
    internal static class PipeIPC
    {
        public const string PIPE_NAME_1 = "3502BD5F-4ABD-49A5-87AC-F655C1984594";
        public const string PIPE_NAME_2 = "C486849A-46C0-4A71-ADB1-CCD7F0A35A27";

        /// <summary>
        /// 
        /// </summary>
        internal static class Server__in
        {
            /// <summary>
            /// 
            /// </summary>
            public delegate void ReceivedInputParamsArrayEventHandler( string data );

            public static event ReceivedInputParamsArrayEventHandler ReceivedData;

            public static void RunListener( string pipeName, CancellationToken ct = default ) //CancellationTokenSource cts = null )
            {
                //var ct = (cts?.Token).GetValueOrDefault( CancellationToken.None );

                Task.Run( async () =>
                {
                    for (; ; )
                    {
                        using ( var pipeServer = new NamedPipeServerStream( pipeName, PipeDirection.In ) )
                        {
#if DEBUG
                            Debug.WriteLine( $"[SERVER] WaitForConnection... (Current TransmissionMode: '{pipeServer.TransmissionMode}')" );
#endif
                            try
                            {
                                await pipeServer.WaitForConnectionAsync( ct ).ConfigureAwait( false );

                                // Read user input and send that to the client process.
                                using ( var sr = new StreamReader( pipeServer ) )
                                {
                                    var data = await sr.ReadLineAsync().ConfigureAwait( false );
#if DEBUG
                                    Debug.WriteLine( $"[SERVER] Read from [CLIENT]: '{data}'." ); Debug.WriteLine( string.Empty );
#endif
                                    ReceivedData?.Invoke( data );
                                }
                            }                            
                            catch ( Exception ex ) // Catch the IOException that is raised if the pipe is broken or disconnected.
                            {
                                Debug.WriteLine( $"[SERVER] Error: '{ex.Message}'" ); Debug.WriteLine( string.Empty );
                            }
                        }
                    }
                }, ct );
            }
            public static async Task RunListenerAsync( string pipeName, CancellationToken ct = default ) //CancellationTokenSource cts = null )
            {
                //var ct = (cts?.Token).GetValueOrDefault( CancellationToken.None );

                var serverTask = Task.Run( async () =>
                {
                    for (; !ct.IsCancellationRequested; )
                    {
                        using ( var pipeServer = new NamedPipeServerStream( pipeName, PipeDirection.In ) )
                        {
#if DEBUG
                            Debug.WriteLine( $"[SERVER] WaitForConnection... (Current TransmissionMode: '{pipeServer.TransmissionMode}')" );
#endif
                            try
                            {
                                await pipeServer.WaitForConnectionAsync( ct ).ConfigureAwait( false );

                                // Read user input and send that to the client process.
                                using ( var sr = new StreamReader( pipeServer ) )
                                {
                                    var data = await sr.ReadLineAsync().ConfigureAwait( false );
#if DEBUG
                                    Debug.WriteLine( $"[SERVER] Read from [CLIENT]: '{data}'." ); Debug.WriteLine( string.Empty );
#endif
                                    ReceivedData?.Invoke( data );
                                }
                            }
                            catch ( Exception ex ) // Catch the IOException that is raised if the pipe is broken or disconnected.
                            {
                                Debug.WriteLine( $"[SERVER] Error: '{ex.Message}'" ); Debug.WriteLine( string.Empty );
                            }
                        }
                    }
                }, ct );
                var waitCancelTask = Task.Delay( Timeout.Infinite, ct );

                await Task.WhenAny( serverTask, waitCancelTask );
            }
            public static async Task< (string data, Exception ex) > RunDataReceiver( string pipeName, CancellationToken ct = default )
            {
                using ( var pipeServer = new NamedPipeServerStream( pipeName, PipeDirection.In ) )
                {
#if DEBUG
                    Debug.WriteLine( $"[SERVER] WaitForConnection... (Current TransmissionMode: '{pipeServer.TransmissionMode}')" );
#endif
                    try
                    {
                        await pipeServer.WaitForConnectionAsync( ct ).ConfigureAwait( false );

                        // Read user input and send that to the client process.
                        using ( var sr = new StreamReader( pipeServer ) )
                        {
                            var data = await sr.ReadLineAsync().ConfigureAwait( false );
#if DEBUG
                            Debug.WriteLine( $"[SERVER] Read from [CLIENT]: '{data}'." ); Debug.WriteLine( string.Empty );
#endif
                            return (data, default);
                        }
                    }
                    catch ( Exception ex ) // Catch the IOException that is raised if the pipe is broken or disconnected.
                    {
                        Debug.WriteLine( $"[SERVER] Error: '{ex.Message}'" ); Debug.WriteLine( string.Empty );
                        return (default, ex);
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        internal static class Client__out
        {
            public static void Send( string pipeName, string data, int connectMillisecondsTimeout = 5_000 )
            {
                using ( var pipeClient = new NamedPipeClientStream( ".", pipeName, PipeDirection.Out ) )
                {
                    pipeClient.ConnectAsync( connectMillisecondsTimeout ).Wait( connectMillisecondsTimeout );
#if DEBUG
                    Debug.WriteLine( $"[CLIENT] Current TransmissionMode: '{pipeClient.TransmissionMode}'" ); 
#endif
                    using ( var sw = new StreamWriter( pipeClient ) { AutoFlush = true } )
                    {
                        // Send a 'sync message' and wait for client to receive it.
#if DEBUG
                        Debug.WriteLine( $"[CLIENT] Send: '{data}'" ); 
#endif
                        sw.WriteLine( data );
                        pipeClient.WaitForPipeDrain_NoThrow();
                    }
                }
#if DEBUG
                Debug.WriteLine( "[CLIENT] Client terminating.\r\n" ); 
#endif
            }
            public static async Task SendAsync( string pipeName, string data, int connectMillisecondsTimeout = 5_000 )
            {
                using ( var pipeClient = new NamedPipeClientStream( ".", pipeName, PipeDirection.Out ) )
                {
                    await pipeClient.ConnectAsync( connectMillisecondsTimeout ).ConfigureAwait( false );
#if DEBUG
                    Debug.WriteLine( $"[CLIENT] Current TransmissionMode: '{pipeClient.TransmissionMode}'" ); 
#endif
                    using ( var sw = new StreamWriter( pipeClient ) { AutoFlush = true } )
                    {
                        // Send a 'sync message' and wait for client to receive it.
#if DEBUG
                        Debug.WriteLine( $"[CLIENT] Send: '{data}'" ); 
#endif
                        await sw.WriteLineAsync( data ).ConfigureAwait( false );
                        pipeClient.WaitForPipeDrain_NoThrow();
                    }
                }
#if DEBUG
                Debug.WriteLine( "[CLIENT] Client terminating.\r\n" ); 
#endif
            }
        }

        private static void WaitForPipeDrain_NoThrow( this NamedPipeClientStream pipeClient )
        {
            try
            {
#pragma warning disable CA1416 // Validate platform compatibility
                pipeClient.WaitForPipeDrain();
#pragma warning restore CA1416 // Validate platform compatibility
            }
            catch ( Exception ex )
            {
                Debug.WriteLine( ex );
            }
        }
    }
}


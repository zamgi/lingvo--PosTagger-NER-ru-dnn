using System;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

using Newtonsoft.Json;

using JsonFormatting = Newtonsoft.Json.Formatting;
using _JsonSerializer_ = System.Text.Json.JsonSerializer;
using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.WebService
{
    /// <summary>
    /// 
    /// </summary>
    internal sealed class PosTaggerWebServiceClient
    {
        private HttpClient _HttpClient;
        private Uri _RunUrl;
        private JsonSerializerOptions _JsonSerializerOptions;
        public PosTaggerWebServiceClient( HttpClient httpClient, string baseUrl )
        {
            _HttpClient = httpClient;

            baseUrl = baseUrl.TrimEnd( '/' );
            _RunUrl = new Uri( $"{baseUrl}/{WebApiConsts.PosTagger.RoutePrefix}/{WebApiConsts.PosTagger.Run}" );

            _JsonSerializerOptions = new JsonSerializerOptions( JsonSerializerDefaults.General ) { IgnoreNullValues = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase, };
            _JsonSerializerOptions.Converters.Add( new JsonStringEnumConverter() );
        }

        public Task< ResultVM > Run( string text, CancellationToken ct ) => Run( new ParamsVM() { Text = text }, ct );
        public async Task< ResultVM > Run( ParamsVM m, CancellationToken ct )
        {
            using ( var jsonContent = m.ToJsonStringContent() )
            using ( var responseMsg = await _HttpClient.GetPostAsync( _RunUrl, jsonContent, ct ).CAX() )
            {
                responseMsg.EnsureSuccessStatusCode();

                var json = await responseMsg.Content.ReadAsStringAsync().CAX();
                //var r = json.FromJSON< ResultVM >();
                var result = _JsonSerializer_.Deserialize< ResultVM >( json, _JsonSerializerOptions );
                return (result);
            }
        }
    }

    /// <summary>
    /// 
    /// </summary>
    internal static class HttpExtensions
    {
        private const string MIMETYPE_APPLICATION_JSON = "application/json";

        [M(O.AggressiveInlining)] public static ConfiguredTaskAwaitable< T > CAX< T >( this Task< T > task ) => task.ConfigureAwait( false );
        [M(O.AggressiveInlining)] public static ConfiguredTaskAwaitable CAX( this Task task ) => task.ConfigureAwait( false );

        [M(O.AggressiveInlining)] public static string ToJSON< T >( this T t, JsonFormatting formatting = JsonFormatting.None ) where T : class => JsonConvert.SerializeObject( t, formatting );
        [M(O.AggressiveInlining)] public static string ToJSON< T >( ref this T t, JsonFormatting formatting = JsonFormatting.None ) where T : struct => JsonConvert.SerializeObject( t, formatting );
        [M(O.AggressiveInlining)] public static T FromJSON< T >( this string json ) => JsonConvert.DeserializeObject< T >( json );

        [M(O.AggressiveInlining)] private static StringContent ToJsonStringContent( this string json ) => new StringContent( json, Encoding.UTF8, MIMETYPE_APPLICATION_JSON );
        [M(O.AggressiveInlining)] public static StringContent ToJsonStringContent< T >( this T t ) where T : class => t.ToJSON().ToJsonStringContent();
        [M(O.AggressiveInlining)] public static StringContent ToJsonStringContent< T >( ref this T t ) where T : struct => t.ToJSON().ToJsonStringContent();

        [M(O.AggressiveInlining)]
        public static async Task< HttpResponseMessage > GetPostAsync( this HttpClient httpClient, string requestUri, CancellationToken ct, HttpCompletionOption hco = HttpCompletionOption.ResponseContentRead ) //=> httpClient.SendAsync( HttpMethod.Post, requestUri, ct );
        {
            using ( var requestMsg = new HttpRequestMessage( HttpMethod.Post, requestUri ) )
            {
                return (await httpClient.SendAsync( requestMsg, hco, ct ).CAX());
            }
        }
        [M(O.AggressiveInlining)]
        public static async Task< HttpResponseMessage > GetPostAsync( this HttpClient httpClient, string requestUri, HttpContent content, CancellationToken ct, HttpCompletionOption hco = HttpCompletionOption.ResponseContentRead ) //=> httpClient.SendAsync( HttpMethod.Post, requestUri, ct, content );
        {
            using ( var requestMsg = new HttpRequestMessage( HttpMethod.Post, requestUri ) { Content = content } )
            {
                return (await httpClient.SendAsync( requestMsg, hco, ct ).CAX());
            }
        }
        [M(O.AggressiveInlining)]
        public static async Task< HttpResponseMessage > GetPostAsync( this HttpClient httpClient, Uri requestUri, CancellationToken ct, HttpCompletionOption hco = HttpCompletionOption.ResponseContentRead )
        {
            using ( var requestMsg = new HttpRequestMessage( HttpMethod.Post, requestUri ) )
            {
                return (await httpClient.SendAsync( requestMsg, hco, ct ).CAX());
            }
        }
        [M(O.AggressiveInlining)]
        public static async Task< HttpResponseMessage > GetPostAsync( this HttpClient httpClient, Uri requestUri, HttpContent content, CancellationToken ct, HttpCompletionOption hco = HttpCompletionOption.ResponseContentRead )
        {
            using ( var requestMsg = new HttpRequestMessage( HttpMethod.Post, requestUri ) { Content = content } )
            {
                return (await httpClient.SendAsync( requestMsg, hco, ct ).CAX());
            }
        }

        [M(O.AggressiveInlining)]
        public static async Task MakePostAsync( this HttpClient httpClient, string requestUri, CancellationToken ct, HttpCompletionOption hco = HttpCompletionOption.ResponseContentRead )
        {
            using ( var requestMsg = new HttpRequestMessage( HttpMethod.Post, requestUri ) )
            using ( var responseMsg = await httpClient.SendAsync( requestMsg, hco, ct ).CAX() )
            {
                responseMsg.EnsureSuccessStatusCode();
            }
        }
        [M(O.AggressiveInlining)]
        public static async Task MakePostAsync( this HttpClient httpClient, string requestUri, HttpContent content, CancellationToken ct, HttpCompletionOption hco = HttpCompletionOption.ResponseContentRead ) //=> httpClient.SendAsync( HttpMethod.Post, requestUri, ct, content );
        {
            using ( var requestMsg = new HttpRequestMessage( HttpMethod.Post, requestUri ) { Content = content } )
            using ( var responseMsg = await httpClient.SendAsync( requestMsg, hco, ct ).CAX() )
            {
                responseMsg.EnsureSuccessStatusCode();
            }
        }
        [M(O.AggressiveInlining)]
        public static async Task MakePostAsync( this HttpClient httpClient, Uri requestUri, CancellationToken ct, HttpCompletionOption hco = HttpCompletionOption.ResponseContentRead )
        {
            using ( var requestMsg = new HttpRequestMessage( HttpMethod.Post, requestUri ) )
            using ( var responseMsg = await httpClient.SendAsync( requestMsg, hco, ct ).CAX() )
            {
                responseMsg.EnsureSuccessStatusCode();
            }
        }
        [M(O.AggressiveInlining)]
        public static async Task MakePostAsync( this HttpClient httpClient, Uri requestUri, HttpContent content, CancellationToken ct, HttpCompletionOption hco = HttpCompletionOption.ResponseContentRead ) //=> httpClient.SendAsync( HttpMethod.Post, requestUri, ct, content );
        {
            using ( var requestMsg = new HttpRequestMessage( HttpMethod.Post, requestUri ) { Content = content } )
            using ( var responseMsg = await httpClient.SendAsync( requestMsg, hco, ct ).CAX() )
            {
                responseMsg.EnsureSuccessStatusCode();
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.Threading.Tasks;

using Microsoft.AspNetCore.Mvc;
#if DEBUG
using Microsoft.Extensions.Logging;
#endif

namespace Lingvo.PosTagger.WebService.Controllers
{
    /// <summary>
    /// 
    /// </summary>
    [ApiController, Route("[controller]")]
    public sealed class PosTaggerController : ControllerBase
    {
        #region [.ctor().]
        private readonly ConcurrentFactory _ConcurrentFactory;
#if DEBUG
        private readonly ILogger< PosTaggerController > _Logger;
        public PosTaggerController( ConcurrentFactory concurrentFactory, ILogger< PosTaggerController > logger )
        {
            _ConcurrentFactory = concurrentFactory;
            _Logger            = logger;
        }
#else
        public PosTaggerController( ConcurrentFactory concurrentFactory ) => _ConcurrentFactory = concurrentFactory;
#endif
        #endregion

        private static ResultVM EMPTY_Result = new ResultVM() { Sents = new List< ResultVM.SentVM >() };
        [HttpPost, Route(WebApiConsts.PosTagger.Run)] public async Task< IActionResult > Run( ParamsVM m )
        {
            try
            {
                await _ConcurrentFactory.LogToFile( m.Text );
#if DEBUG
                _Logger.LogInformation( $"start '{m.ToText()}'..." );
#endif
                ResultVM result;
                if ( !m.Text.IsNullOrWhiteSpace() )
                {
                    var t = await _ConcurrentFactory.TryRunAsync( m.Text, m.ModelType, getFirstModelTypeIfMissing: true ).CAX();
                    result = t.ToResultVM();
                }
                else
                {
                    result = EMPTY_Result;
                }
#if DEBUG
                _Logger.LogInformation( $"end '{m.ToText()}'." );
#endif
                return Ok( result );
            }
            catch ( Exception ex )
            {
                return Ok( ex.ToErrorVM() );
            }
        }

        [HttpGet, Route(WebApiConsts.PosTagger.GetModelInfoKeys)] public IActionResult GetModelInfoKeys( string regimenModelType )
        {
            try
            {
                return Ok( _ConcurrentFactory.GetModelInfoKeys( regimenModelType ) );
            }
            catch ( Exception ex )
            {
                return Ok( ex.ToErrorVM() );
            }
        }

        [HttpGet, Route(WebApiConsts.PosTagger.Log)] public Task< IActionResult > Log() => ReadLogFile();
        [HttpGet, Route(WebApiConsts.PosTagger.ReadLogFile)] public async Task< IActionResult > ReadLogFile()
        {
            try
            {
                var text = await _ConcurrentFactory.ReadLogFile();
                return Ok( text );
            }
            catch ( Exception ex )
            {
                return Ok( ex.ToErrorVM() );
            }
        }
        [HttpGet, Route(WebApiConsts.PosTagger.DelLog)] public Task< IActionResult > DelLog() => DeleteLogFile();
        [HttpGet, Route(WebApiConsts.PosTagger.DeleteLogFile)] public async Task< IActionResult > DeleteLogFile()
        {
            try
            {
                await _ConcurrentFactory.DeleteLogFile();
                return Ok( "Success" );
            }
            catch ( Exception ex )
            {
                return Ok( ex.ToErrorVM() );
            }
        }        
    }
}

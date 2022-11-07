using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

using Lingvo.PosTagger.Applications;
using Lingvo.PosTagger.Models;
using Lingvo.PosTagger.Utils;
using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Text
{
    /// <summary>
    /// 
    /// </summary>
    public enum TooLongSequence
    {
        Ignore,
        Truncation
    }

    /// <summary>
    /// 
    /// </summary>
    public class Corpus : IDisposable
    {
        /// <summary>
        /// 
        /// </summary>
        protected struct offset_t
        {
            public long src_offset;
            public long tgt_offset;
            public int  src_length;
            public int  tgt_length;

            [M(O.AggressiveInlining)] public static implicit operator offset_t( in (long src_offset, int src_length, long tgt_offset, int tgt_length) t )
                => new offset_t() { src_offset = t.src_offset, src_length = t.src_length, tgt_offset = t.tgt_offset, tgt_length = t.tgt_length };
        }  

        /// <summary>
        /// 
        /// </summary>
        unsafe private struct FileStreamHolder : IDisposable
        {
            private FileStream _Fs;
            private Decoder _Decoder;
            private GCHandle _BytesBuf_GCHandle;
            private byte* _BytesBuf;
            private GCHandle _CharsBuf_GCHandle;
            private char* _CharsBuf;
            private int _CharsBuf_MaxLen;
            public FileStreamHolder( string fileName, Decoder decoder = null, int max_text_line_len = 0x1000 )
            {
                _Fs = File.OpenRead( fileName );
                _Decoder = decoder ?? Encoding.UTF8.GetDecoder();

                var bytes = new byte[ 4 * max_text_line_len ];
                var chars = new char[ max_text_line_len ];

                _BytesBuf_GCHandle = GCHandle.Alloc( bytes, GCHandleType.Pinned );
                _BytesBuf = (byte*) _BytesBuf_GCHandle.AddrOfPinnedObject();

                _CharsBuf_GCHandle = GCHandle.Alloc( chars, GCHandleType.Pinned );
                _CharsBuf = (char*) _CharsBuf_GCHandle.AddrOfPinnedObject();
                _CharsBuf_MaxLen = max_text_line_len;
            }
            public string ReadLine( long offset, int length )
            {
                _Fs.Position = offset;

                var bytes_span = new Span< byte >( _BytesBuf, length );
                var read_chars_count = _Fs.Read( bytes_span );

                var chars_span = new Span< char >( _CharsBuf, _CharsBuf_MaxLen );
                var chars_count = _Decoder.GetChars( bytes_span, chars_span, flush: true );

                var line = new string( _CharsBuf, 0, chars_count );
                return (line);
            }
            public int Read( long offset, int length )
            {
                _Fs.Position = offset;

                var bytes_span = new Span<byte>( _BytesBuf, length );
                var read_chars_count = _Fs.Read( bytes_span );

                var chars_span = new Span<char>( _CharsBuf, _CharsBuf_MaxLen );
                var chars_count = _Decoder.GetChars( bytes_span, chars_span, flush: true );

                return (chars_count);
            }
            public char* Chars => _CharsBuf;
            //public string ReadLine( in (long offset, int length) t )
            //{
            //    _Fs.Position = t.offset;

            //    var bytes_span = new Span< byte >( _BytesBuf, t.length );
            //    var read_chars_count = _Fs.Read( bytes_span );

            //    var chars_span = new Span< char >( _CharsBuf, _CharsBuf_MaxLen );
            //    var chars_count = _Decoder.GetChars( bytes_span, chars_span, flush: true );

            //    var line = new string( _CharsBuf, 0, chars_count );
            //    return (line);
            //}
            public void Dispose()
            {
                _Fs.Dispose();
                _BytesBuf_GCHandle.Free();
                _CharsBuf_GCHandle.Free();
            }
        }

        private bool              _ShowTokenDist   = true;
        private TooLongSequence   _TooLongSequence;
        private CancellationToken _Ct;

        protected int              _MaxSrcSentLength;
        protected int              _MaxTgtSentLength;
        protected int              _BatchSize;
        protected string           _SrcFilePath;
        protected string           _TgtFilePath;
        protected List< offset_t > _OffsetMap;
        protected bool _UniqueTempFileNames;

        public Corpus( string corpusFilePath, int batchSize, int maxSentLength, TooLongSequence tooLongSequence
                     , CancellationToken cancellationToken = default
                     , bool uniqueTempFileNames = true )
        {
            Logger.WriteLine( $"Loading sequence labeling corpus from '{corpusFilePath}' MaxSentLength = '{maxSentLength}'" );
            _TooLongSequence     = tooLongSequence;
            _Ct                  = cancellationToken;
            _BatchSize           = Math.Max( 1, batchSize );
            _MaxSrcSentLength    = maxSentLength;
            _MaxTgtSentLength    = maxSentLength;
            _UniqueTempFileNames = uniqueTempFileNames;

            (_SrcFilePath, _TgtFilePath, _OffsetMap) = ConvertTrainFile2SeqLabelFormat( corpusFilePath, uniqueTempFileNames, cancellationToken );

            //Test_1( _SrcFilePath, _TgtFilePath, _OffsetMap );
            //Test_2( _SrcFilePath, _TgtFilePath, _OffsetMap );
        }
        public void Dispose()
        {
            File_Delete_NoThrow( _SrcFilePath );
            File_Delete_NoThrow( _TgtFilePath );
        }

        public int BatchSize => _BatchSize;

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        public (Vocab srcVocab, Vocab tgtVocab) BuildVocabs( bool vocabIgnoreCase, int vocabSize )
        {
            Logger.WriteLine( $"Start build vocabs." );

            var bilder = new CorpusBatchBuilder();
            foreach ( var batch in this.GetBatchs( shuffle: false ) )
            {
                bilder.CountSntPairTokens( batch.SntPairs );
            }

            var (srcVocab, tgtVocab) = bilder.GenerateVocabs( vocabIgnoreCase, vocabSize );
            
            Logger.WriteLine( $"End build vocabs." );
            _Ct.ThrowIfCancellationRequested();

            return (srcVocab, tgtVocab);
        }
        public Vocab BuildTargetVocab( bool vocabIgnoreCase, int vocabSize )
        {
            Logger.WriteLine( $"Start build tgt-vocab." );

            var cbb = new CorpusBatchBuilder();
            foreach ( var batch in this.GetBatchs( shuffle: false ) )
            {
                cbb.CountTargetTokens( batch.SntPairs );
            }

            var tgtVocab = cbb.GenerateTargetVocab( vocabIgnoreCase, vocabSize );
            Logger.WriteLine( $"End build tgt-vocab." );

            _Ct.ThrowIfCancellationRequested();

            return (tgtVocab);
        }

        private static void Shuffle< T >( List< T > lst )
        {
            var rnd = new Random();
            for ( int i = 0, cnt = lst.Count; i < cnt; i++ )
            {
                var idx = rnd.Next( 0, cnt );
                var t = lst[ i ];
                lst[ i ] = lst[ idx ];
                lst[ idx ] = t;
            }
        }
        private (string srcShuffledFilePath, string tgtShuffledFilePath) ShuffleAll( bool showTokenDist )
        {
            var title = Misc.Console_Title();

            #region [.showTokenDist.]
            var dictSrcLenDist = showTokenDist ? new SortedDictionary< int, int >() : null;
            var dictTgtLenDist = showTokenDist ? new SortedDictionary< int, int >() : null;
            #endregion

            var tmp_dir = GetTempWorkDirPath();
            var srcShuffledFilePath = Path.Combine( tmp_dir, $"__(txt)_shuffled__{GetUniqueTempFileNames()}.tmp" );
            var tgtShuffledFilePath = Path.Combine( tmp_dir, $"__(lbl)_shuffled__{GetUniqueTempFileNames()}.tmp" );
            if ( !Directory.Exists( tmp_dir ) ) Directory.CreateDirectory( tmp_dir );

            Logger.WriteLine( $"Shuffling corpus from temp-files." );

            var tooLongSrcSntCnt   = 0;
            var tooLongTgtSntCnt   = 0;
            var truncateTooLongSeq = (_TooLongSequence == TooLongSequence.Truncation);

            Shuffle( _OffsetMap );

            using ( var src_sr = new FileStreamHolder( _SrcFilePath ) )
            using ( var tgt_sr = new FileStreamHolder( _TgtFilePath ) )
            using ( var src_sw = new StreamWriter( srcShuffledFilePath, append: false, Encoding.UTF8 ) )
            using ( var tgt_sw = new StreamWriter( tgtShuffledFilePath, append: false, Encoding.UTF8 ) )
            {
                for ( int sentIdx = 0, sentCnt = _OffsetMap.Count; sentIdx < sentCnt; sentIdx++ )
                {
                    var o = _OffsetMap[ sentIdx ];

                    var src = src_sr.ReadLine( o.src_offset, o.src_length );
                    var tgt = tgt_sr.ReadLine( o.tgt_offset, o.tgt_length );
                    if ( src.IsNullOrEmpty() && tgt.IsNullOrEmpty() )
                    {
                        continue;
                    }

                    var sp = new RawSntPair( src, tgt, _MaxSrcSentLength, _MaxTgtSentLength, truncateTooLongSeq );

                    #region [.showTokenDist.]
                    if ( showTokenDist )
                    {
                        var si = sp.SrcLength / 100;
                        if ( !dictSrcLenDist.TryGetValue( si, out var cnt ) )
                        {
                            dictSrcLenDist.Add( si, 1 );
                        }
                        else
                        {
                            dictSrcLenDist[ si ] = cnt + 1;
                        }

                        var ti = sp.TgtLength / 100;
                        if ( !dictTgtLenDist.TryGetValue( ti, out cnt ) )
                        {
                            dictTgtLenDist.Add( ti, 1 );
                        }
                        else
                        {
                            dictTgtLenDist[ ti ] = cnt + 1;
                        }
                    }
                    #endregion

                    #region [.isTooLongSent.]
                    if ( _MaxSrcSentLength < sp.SrcLength ) { tooLongSrcSntCnt++; continue; }
                    if ( _MaxTgtSentLength < sp.TgtLength ) { tooLongTgtSntCnt++; continue; }
                    #endregion

                    src_sw.WriteLine( sp.SrcSnt );
                    tgt_sw.WriteLine( sp.TgtSnt );

                    var sentNum = sentIdx + 1;
                    if ( (sentNum % 10_000) == 0 )
                    {
                        if ( _Ct.IsCancellationRequested ) return (srcShuffledFilePath, tgtShuffledFilePath); //break;
                        Misc.Console_Title( $"shuffle sents: {sentNum:#,#}..." );
                    }
                }
            }

            Misc.Console_Title( $"shuffle sents: {_OffsetMap.Count:#,#}..." );
            Logger.WriteLine( $"Shuffled '{_OffsetMap.Count:#,#}' sentence pairs{((0 < tooLongSrcSntCnt + tooLongTgtSntCnt) ? $" (total-sent-count: {(_OffsetMap.Count + tooLongSrcSntCnt + tooLongTgtSntCnt):#,#})" : null)}." );
            Misc.Console_Title( title );

            if ( 0 < tooLongSrcSntCnt ) Logger.WriteWarnLine( $"Found {tooLongSrcSntCnt} source sentences are longer than '{_MaxSrcSentLength}' tokens, ignore them." );
            if ( 0 < tooLongTgtSntCnt ) Logger.WriteWarnLine( $"Found {tooLongTgtSntCnt} target sentences are longer than '{_MaxTgtSentLength}' tokens, ignore them." );

            #region [.showTokenDist.]
            if ( showTokenDist )
            {
                var buf = new StringBuilder( 0x100 );
                foreach ( var (len, cnt) in dictSrcLenDist.OrderByDescending( p => p.Value ) )
                {
                    if ( buf.Length != 0 ) buf.Append( ", " );
                    buf.Append( $"/{len * 100} ~ {(len + 1) * 100}: {cnt}/" );
                }
                buf.Append( '.' ).Insert( 0, "Src token length distribution: " );
                Logger.WriteLine( buf.ToString() );

                buf.Clear();
                foreach ( var (len, cnt) in dictTgtLenDist.OrderByDescending( p => p.Value ) )
                {
                    if ( buf.Length != 0 ) buf.Append( ", " );
                     buf.Append( $"/{len * 100} ~ {(len + 1) * 100}: {cnt}/" );
                }
                buf.Append( '.' ).Insert( 0, $"Tgt token length distribution: " );
                Logger.WriteLine( buf.ToString() );
            }
            #endregion

            return (srcShuffledFilePath, tgtShuffledFilePath);
        }

        public IEnumerable< CorpusBatch > GetBatchs( bool shuffle /*= false*/ )
        {
            #region [.shuffle.]
            string srcShuffledFilePath, tgtShuffledFilePath;
            if ( shuffle )
            {
                (srcShuffledFilePath, tgtShuffledFilePath) = ShuffleAll( _ShowTokenDist ); _ShowTokenDist = false;
            }
            else
            {
                (srcShuffledFilePath, tgtShuffledFilePath) = (_SrcFilePath, _TgtFilePath);
            }
            #endregion

            try
            {
                using ( var src_sr = new StreamReader( srcShuffledFilePath ) )
                using ( var tgt_sr = new StreamReader( tgtShuffledFilePath ) )
                {
                    var buf = new List< SntPair >( 10_000 );

                    [M(O.AggressiveInlining)] bool try_read_SntPair( out SntPair sp )
                    {
                        var src_line = src_sr.ReadLine();
                        var tgt_line = tgt_sr.ReadLine();
                        if ( (src_line == null) || (tgt_line == null) )
                        {
                            sp = default;
                            return (false);
                        }

                        src_line = src_line.Trim();
                        tgt_line = tgt_line.Trim();
                        sp = new SntPair( src_line, tgt_line );
                        return (true);
                    };
                    #region comm.
                    //[M(O.AggressiveInlining)] IEnumerable< CorpusBatch > get_CorpusBatch()
                    //{
                    //    for ( var i = 0; (i < buf.Count) && !_Ct.IsCancellationRequested; i += _BatchSize )
                    //    {
                    //        var size  = Math.Min( _BatchSize, buf.Count - i );
                    //        var batch = new CorpusBatch( buf.GetRange( i, size ) );
                    //        yield return (batch);
                    //    }
                    //}; 
                    #endregion

                    for ( ; !_Ct.IsCancellationRequested; )
                    {
                        if ( _BatchSize == buf.Count )
                        {
                            #region comm.
                            //foreach ( var batch in get_CorpusBatch() )
                            //{
                            //    yield return (batch);
                            //} 
                            #endregion
                            var sps = buf.ToList( _BatchSize ); //buf.GetRange( 0, _BatchSize ); 
                            var batch = new CorpusBatch( sps );
                            yield return (batch);
                            buf.Clear();
                        }

                        if ( !try_read_SntPair( out var sp ) )
                        {
                            break;
                        }

                        buf.Add( sp );
                    }

                    #region comm.
                    //foreach ( var batch in get_CorpusBatch() )
                    //{
                    //    yield return (batch);
                    //} 
                    #endregion
                    if ( !_Ct.IsCancellationRequested && (0 < buf.Count) )
                    {
                        var batch = new CorpusBatch( buf );
                        yield return (batch);
                    }
                }
            }
            finally
            {
                if ( shuffle )
                {
                    File_Delete_NoThrow( srcShuffledFilePath );
                    File_Delete_NoThrow( tgtShuffledFilePath );
                }
            }
        }

        private string GetUniqueTempFileNames() => GetUniqueTempFileNames( _UniqueTempFileNames );
        private static string GetUniqueTempFileNames( bool uniqueTempFileNames ) => uniqueTempFileNames ? Path.GetRandomFileName() : null;
        private static string GetTempWorkDirPath() => Path.Combine( Directory.GetCurrentDirectory(), ".tmp" );
        private static void CreateOrClearTmpDirectory( string tmp_dir, bool uniqueTempFileNames )
        {
            if ( !Directory.Exists( tmp_dir ) )
            {
                Directory.CreateDirectory( tmp_dir );
            }
            else if ( !uniqueTempFileNames )
            {
                foreach ( var dir in Directory.GetDirectories( tmp_dir, "*", SearchOption.AllDirectories ) )
                {
                    if ( Directory.Exists( dir ) )
                    {
                        try { Directory.Delete( dir, true ); } catch ( Exception ex ) { Console.WriteLine( ex ); }
                    }
                }
                foreach ( var fn in Directory.EnumerateFiles( tmp_dir, "*", SearchOption.AllDirectories ) )
                {
                    File_Delete_NoThrow( fn );
                }
            }
        }
        protected static void File_Delete_NoThrow( string fn )
        {
            try
            {
                File.Delete( fn );
            }
            catch
            {
                ;
            }
        }
        unsafe private static (string srcFilePath, string tgtFilePath, List< offset_t > offsetMap) ConvertTrainFile2SeqLabelFormat( string filePath, bool uniqueTempFileNames, CancellationToken ct = default )
        {
            var title = Misc.Console_Title();
            Logger.WriteLine( $"Start convert sequence labeling corpus file '{filePath}' to parallel corpus format to temp-files." );

            var separator = new char[] { ' ', '\t' };
            var currSent  = new List< (string src, string tgt) >();

            var srcFilePath = default(string);
            var tgtFilePath = default(string);
            var offsetMap   = default(List< offset_t >);

            var sentCount = 0;
            using ( var sr = new StreamReader( filePath ) )
            {
                var tmp_dir = GetTempWorkDirPath();
                srcFilePath = Path.Combine( tmp_dir, $"__(txt)__{GetUniqueTempFileNames( uniqueTempFileNames )}.tmp" );
                tgtFilePath = Path.Combine( tmp_dir, $"__(lbl)__{GetUniqueTempFileNames( uniqueTempFileNames )}.tmp" );
                CreateOrClearTmpDirectory( tmp_dir, uniqueTempFileNames );

                var fuzzy_sentCount = (int) (sr.BaseStream.Length / 0x100);
                offsetMap = new List< offset_t >( fuzzy_sentCount );

                using var src_stream = File.OpenWrite( srcFilePath ); 
                using var tgt_stream = File.OpenWrite( tgtFilePath );

                var encoder         = Encoding.UTF8.GetEncoder();
                var crlf_bytes      = Encoding.UTF8.GetBytes( Environment.NewLine );
                var crlf_bytes_len  = crlf_bytes.Length;
                var space_bytes     = new[] { (byte) ' ' }; //Encoding.UTF8.GetBytes( new[] { ' ' } );
                var space_bytes_len = space_bytes.Length;
                long src_pos        = 0L;
                long tgt_pos        = 0L;
                const int BUF_LEN   = 0x1000;
                var  buf            = stackalloc byte[ BUF_LEN ];

                [M(O.AggressiveInlining)] void write_currSent()
                {
                    //var src_pos = src_stream.Position;
                    //var tgt_pos = tgt_stream.Position;
                    var src_len = 0;
                    var tgt_len = 0;

                    for ( int i = 0, len = currSent.Count - 1; i <= len; i++ )
                    {
                        var (src, tgt) = currSent[ i ];

                        fixed ( char* ptr = src )
                        {
                            var bytes_count = encoder.GetBytes( ptr, src.Length, buf, BUF_LEN, flush: true );
                            src_stream.Write( new ReadOnlySpan< byte >( buf, bytes_count ) ); src_len += bytes_count;
                        }
                        fixed ( char* ptr = tgt )
                        {
                            var bytes_count = encoder.GetBytes( ptr, tgt.Length, buf, BUF_LEN, flush: true );
                            tgt_stream.Write( new ReadOnlySpan< byte >( buf, bytes_count ) ); tgt_len += bytes_count;
                        }

                        // src_sw.Write( src ); src_len += src.Length;
                        // tgt_sw.Write( tgt ); tgt_len += tgt.Length;

                        if ( i != len )
                        {
                            src_stream.Write( space_bytes ); src_len += space_bytes_len; // src_sw.Write( ' ' ); src_len++;
                            tgt_stream.Write( space_bytes ); tgt_len += space_bytes_len; // tgt_sw.Write( ' ' ); tgt_len++;
                        }
                    }

                    offsetMap.Add( (src_pos, src_len/*(int) (src_stream.Position - src_pos)*/, tgt_pos, tgt_len/*(int) (tgt_stream.Position - tgt_pos)*/) );

                    src_stream.Write( crlf_bytes ); // src_sw.WriteLine();
                    tgt_stream.Write( crlf_bytes ); // tgt_sw.WriteLine();

                    src_pos += src_len + crlf_bytes_len;
                    tgt_pos += tgt_len + crlf_bytes_len;
                };
                
                for ( var line = sr.ReadLine(); line != null; line = sr.ReadLine() )
                {
                    if ( line.IsNullOrWhiteSpace() )
                    {
                        if ( 0 < currSent.Count )
                        {
                            if ( (++sentCount % 10_000) == 0 )
                            {
                                if ( ct.IsCancellationRequested ) break;
                                Misc.Console_Title( $"read sents: {sentCount:#,#}..." );
                            }

                            write_currSent();
                            currSent.Clear();
                        }
                    }
                    else
                    {
                        var array = line.Split( separator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries );
                        if ( array.Length < 2 ) continue;
                        var src = array[ 0 ];
                        var tgt = array[ 1 ];

                        currSent.Add( (src, tgt) );
                    }
                }
                if ( 0 < currSent.Count )
                {
                    write_currSent();
                }
                Misc.Console_Title( $"read sents: {sentCount:#,#}..." );
            }

            Logger.WriteLine( $"End convert sequence labeling corpus file '{filePath}' to parallel corpus format to temp-files (sents-count: {sentCount:#,#})." );
            Misc.Console_Title( title );

            return (srcFilePath, tgtFilePath, offsetMap);
        }

#if DEBUG
        unsafe private static void Test_1( string srcFilePath, string tgtFilePath, List< offset_t > offsetMap )
        {
            var lst = new List< (offset_t _, string src_line, string tgt_line) >( offsetMap.Count );
            using ( var e = offsetMap.GetEnumerator() )
            using ( var src_e = File.ReadLines( srcFilePath ).GetEnumerator() )
            using ( var tgt_e = File.ReadLines( tgtFilePath ).GetEnumerator() )
            {
                for( ; src_e.MoveNext() && tgt_e.MoveNext() && e.MoveNext(); )
                {
                    lst.Add( (e.Current, src_e.Current, tgt_e.Current) );
                }
            }

            Shuffle( lst );
            //---------------------------------------//


            //var max_len = offsetMap.Max( t => Math.Max( t.src_length, t.tgt_length ) );
            const int max_len = 0x1000;
            var buf        = stackalloc byte[ max_len ];
            var chars      = stackalloc char[ max_len ];
            var chars_span = new Span< char >( chars, max_len );
            var decoder    = Encoding.UTF8.GetDecoder();

            [M(O.AggressiveInlining)] static bool is_equal( string s, char* ptr, int len )
            {
                if ( s.Length == len )
                {
                    for ( var i = 0; i < len; i++ )
                    {
                        if ( s[ i ] != *ptr++ )
                        {
                            return (false);
                        }
                    }
                    return (true);
                }
                return (false);
            };

            using ( var src_fs = File.OpenRead( srcFilePath ) )
            using ( var tgt_fs = File.OpenRead( tgtFilePath ) )
            {
                foreach ( var t in lst )
                {
        #region [.src.]
                    src_fs.Position = t._.src_offset;

                    var bytes_span = new Span< byte >( buf, t._.src_length );
                    var read_chars_count = src_fs.Read( bytes_span );

                    var chars_count = decoder.GetChars( bytes_span, chars_span, flush: true );

                    //var src_line = new string( chars, 0, chars_count );
                    //Debug.Assert( src_line == t.src_line );
                    Debug.Assert( is_equal( t.src_line, chars, chars_count ) );
        #endregion

        #region [.tgt.]
                    tgt_fs.Position = t._.tgt_offset;

                    bytes_span = new Span< byte >( buf, t._.tgt_length );
                    read_chars_count = tgt_fs.Read( bytes_span );

                    chars_count = decoder.GetChars( bytes_span, chars_span, flush: true );

                    //var tgt_line = new string( chars, 0, chars_count );
                    //Debug.Assert( tgt_line == t.tgt_line );
                    Debug.Assert( is_equal( t.tgt_line, chars, chars_count ) );
        #endregion
                }
            }
        }
        unsafe private static void Test_2( string srcFilePath, string tgtFilePath, List< offset_t > offsetMap )
        {
            var lst = new List< (offset_t _, string src_line, string tgt_line) >( offsetMap.Count );
            using ( var e = offsetMap.GetEnumerator() )
            using ( var src_e = File.ReadLines( srcFilePath ).GetEnumerator() )
            using ( var tgt_e = File.ReadLines( tgtFilePath ).GetEnumerator() )
            {
                for( ; src_e.MoveNext() && tgt_e.MoveNext() && e.MoveNext(); )
                {
                    lst.Add( (e.Current, src_e.Current, tgt_e.Current) );
                }
            }

            Shuffle( lst );
            //---------------------------------------//

            [M(O.AggressiveInlining)] static bool is_equal( string s, char* ptr, int len )
            {
                if ( s.Length == len )
                {
                    for ( var i = 0; i < len; i++ )
                    {
                        if ( s[ i ] != *ptr++ )
                        {
                            return (false);
                        }
                    }
                    return (true);
                }
                return (false);
            };

            using ( var src_fs = new FileStreamHolder( srcFilePath ) )
            using ( var tgt_fs = new FileStreamHolder( tgtFilePath ) )
            {
                foreach ( var t in lst )
                {
        #region [.src.]
                    var chars_count = src_fs.Read( t._.src_offset, t._.src_length );

                    //var src_line = new string( src_fs.Chars, 0, chars_count );
                    //Debug.Assert( src_line == t.src_line );
                    Debug.Assert( is_equal( t.src_line, src_fs.Chars, chars_count ) );
        #endregion

        #region [.tgt.]
                    chars_count = tgt_fs.Read( t._.tgt_offset, t._.tgt_length );

                    //var tgt_line = new string( tgt_fs.Chars, 0, chars_count );
                    //Debug.Assert( tgt_line == t.tgt_line );
                    Debug.Assert( is_equal( t.tgt_line, tgt_fs.Chars, chars_count ) );
        #endregion
                }
            }
        }
#endif
        public void OptionsWasChanging( in OptionsAllowedChanging opts )
        {
            if ( _BatchSize != opts.BatchSize ) _BatchSize = opts.BatchSize;
        }
    }
}

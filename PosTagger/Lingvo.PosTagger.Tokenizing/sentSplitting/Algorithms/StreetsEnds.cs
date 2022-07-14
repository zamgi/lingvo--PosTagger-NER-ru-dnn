using System.Collections.Generic;
using System.Linq;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.SentSplitting
{
    /// <summary>
    /// 
    /// </summary>
    unsafe internal struct streets_ends_as_end_complex_word_t
    {
        /// <summary>
        /// 
        /// </summary>
        private sealed class StreetEnd_t
        {
            private Map< char, StreetEnd_t > _Prev;
            public bool IsLeaf        { [M(O.AggressiveInlining)] get; private set; }
            public bool OnlyFullValue { [M(O.AggressiveInlining)] get; private set; }
            public bool HasPrev       { [M(O.AggressiveInlining)] get => (_Prev != null); }
            [M(O.AggressiveInlining)] public bool TryGetValue( char key, out StreetEnd_t sp ) => _Prev.TryGetValue( key, out sp );

            public void Add( string value, int startIndex, bool onlyFullValue )
            {
                var sp = this;
                for ( ; 0 <= startIndex; startIndex-- )
                {                    
                    var prev_ch = value[ startIndex ];
                    if ( !sp.HasPrev )
                    {
                        sp._Prev = new Map< char, StreetEnd_t >();

                        var prev_sp = new StreetEnd_t();
                        sp._Prev.Add( prev_ch, prev_sp );
                        sp = prev_sp;
                    }
                    else if ( sp._Prev.TryGetValue( prev_ch, out var exists_sp ) )
                    {
                        sp = exists_sp;
                    }
                    else
                    {
                        var prev_sp = new StreetEnd_t();
                        sp._Prev.Add( prev_ch, prev_sp );
                        sp = prev_sp;
                    }
                }
                sp.IsLeaf        = true;
                sp.OnlyFullValue = onlyFullValue;
            }
#if DEBUG
            private IList< string > AsStrings()
            {
                if ( HasPrev )
                {
                    var lst = new List< string >( _Prev.Count );
                    foreach ( var p in _Prev )
                    {
                        var text = p.Key +
                                   (IsLeaf ? $"(+{(OnlyFullValue ? "[F]" : null)})" : null);
                        var ss = p.Value.AsStrings();
                        foreach ( var s in ss )
                        {
                            lst.Add( s + text );
                        }
                    }
                    return (lst);
                }
                return (new[] { (IsLeaf ? $"(+{(OnlyFullValue ? "[F]" : null)})" : null) });
            }
            public string AsString( char ch ) => string.Join( $"{ch}, ", AsStrings() ) + ch;
            public override string ToString() => string.Join( ", ", AsStrings() );
#endif
        }
        /// <summary>
        /// 
        /// </summary>
        private struct StreetsEnds_t
        {
            public static StreetsEnds_t Create() => new StreetsEnds_t() { _Roots = new Map< char, StreetEnd_t >() };

            private Map< char, StreetEnd_t > _Roots;
            public void Add( string value, bool onlyFullValue = false )
            {
                if ( value.IsNullOrEmpty() ) return;

                var idx = value.Length - 1;
                var ch  = value[ idx ];
                if ( !_Roots.TryGetValue( ch, out var exists_sp ) )
                {
                    exists_sp = new StreetEnd_t();
                    _Roots.Add( ch, exists_sp );
                }
                exists_sp.Add( value, idx - 1, onlyFullValue );
            }
            public bool Has( char* ptr, int len )
            {
                var idx = len - 1;
                var ch  = ptr[ idx ];
                if ( _Roots.TryGetValue( ch, out var exists_sp ) )
                {
                    for ( idx--; ; idx-- )
                    {
                        if ( idx < 0 )
                        {
                            return (exists_sp.IsLeaf);
                        }
                        if ( !exists_sp.HasPrev )
                        {
                            return (!exists_sp.OnlyFullValue && exists_sp.IsLeaf);
                        }

                        ch = ptr[ idx ];
                        if ( !exists_sp.TryGetValue( ch, out var _exists_sp ) )
                        {
                            break;
                        }
                        exists_sp = _exists_sp;
                    }
                }
                return (false);
            }
#if DEBUG
            public override string ToString() => _Roots.Count.ToString();
            public IList< string > AsStrings() => _Roots.Select( p => p.Value.AsString( p.Key ) ).ToArray();
#endif
        }

        private StreetsEnds_t _StreetsEnds;
        private int           _ValuesMinLength;
        public streets_ends_as_end_complex_word_t( IEnumerable< string > streetsEnds )
        {
            _StreetsEnds = StreetsEnds_t.Create();
            var minLen = int.MaxValue;
            foreach ( var streetEnd in streetsEnds )
            {
                _StreetsEnds.Add( streetEnd );

                if ( streetEnd.Length < minLen ) minLen = streetEnd.Length;
            }
            _ValuesMinLength = minLen;
        }
        [M(O.AggressiveInlining)] public bool HasStreetEnds( char* ptr, int len ) => ((_ValuesMinLength < len) && _StreetsEnds.Has( ptr, len ));
        public int ValuesMinLength => _ValuesMinLength;
    }
}

using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace Lingvo.PosTagger.Tensors.Cuda.RuntimeCompiler
{
    /// <summary>
    /// 
    /// </summary>
    public class KernelConfig
    {
        /// <summary>
        /// 
        /// </summary>
        public sealed class EqualityComparer : IEqualityComparer< KernelConfig >
        {
            public static EqualityComparer Inst { get; } = new EqualityComparer();
            private EqualityComparer() { }
            public bool Equals( KernelConfig x, KernelConfig y )
            {
                if ( x.Count != y.Count )
                {
                    return (false);
                }

                foreach ( KeyValuePair<string, string> p in x.AllValues() )
                {
                    if ( !y.TryGetValue( p.Key, out string otherValue ) || (p.Value != otherValue) )
                    {
                        return (false);
                    }
                }
                return (true);
            }
            public int GetHashCode( [DisallowNull] KernelConfig obj )
            {
                var hash = 0;
                foreach ( KeyValuePair<string, string> p in obj.AllValues() )
                {
                    hash ^= p.Key.GetHashCode();
                    hash ^= p.Value.GetHashCode();
                }
                return (hash);
            }
        }
        //public override bool Equals( object obj )
        //{
        //    if ( !(obj is KernelConfig o) )
        //    {
        //        return (false);
        //    }

        //    if ( _Values.Count != o._Values.Count )
        //    {
        //        return (false);
        //    }

        //    foreach ( KeyValuePair<string, string> p in _Values )
        //    {
        //        if ( _Values.TryGetValue( p.Key, out string oValue ) )
        //        {
        //            if ( !p.Value.Equals( oValue ) )
        //            {
        //                return (false);
        //            }
        //        }
        //        else
        //        {
        //            return (false);
        //        }
        //    }
        //    return (true);
        //}
        //public override int GetHashCode()
        //{
        //    var hash = 0;
        //    foreach ( KeyValuePair<string, string> p in _Values )
        //    {
        //        hash ^= p.Key.GetHashCode();
        //        hash ^= p.Value.GetHashCode();
        //    }
        //    return (hash);
        //}

        private readonly SortedDictionary<string, string> _Values;
        public KernelConfig() => _Values = new SortedDictionary<string, string>();

        public IEnumerable<string> Keys => _Values.Keys;
        public IEnumerable<KeyValuePair<string, string>> AllValues() => _Values;
        public int Count => _Values.Count;
        public bool TryGetValue( string name, out string value ) => _Values.TryGetValue( name, out value );
        public bool ContainsKey( string name ) => _Values.ContainsKey( name );
        public void Set( string name, string value ) => _Values[ name ] = value;
        public string ApplyToTemplate( string templateCode )
        {
            var buf = new StringBuilder();
            foreach ( KeyValuePair<string, string> p in _Values )
            {
                buf.AppendFormat( "#define {0} {1}\n", p.Key, p.Value );
            }
            buf.AppendLine( templateCode );
            return (buf.ToString());
        }
    }
}

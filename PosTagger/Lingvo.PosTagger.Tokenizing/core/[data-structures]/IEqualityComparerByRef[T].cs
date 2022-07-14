namespace System.Collections.Generic
{
    /// <summary>
    /// 
    /// </summary>
    public interface IEqualityComparerByRef< T > where T : struct
    {
        bool Equals( in T x, in T y );
        int GetHashCode( in T obj );
    }
}

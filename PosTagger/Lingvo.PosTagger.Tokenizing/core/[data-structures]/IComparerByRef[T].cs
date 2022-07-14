namespace System.Collections.Generic
{
    /// <summary>
    /// 
    /// </summary>
    public interface IComparerByRef< T > where T : struct
    {
        int Compare( in T x, in T y );
    }
}

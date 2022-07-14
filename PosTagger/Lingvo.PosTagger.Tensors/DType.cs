using System;
using System.Runtime.InteropServices;

namespace Lingvo.PosTagger.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public enum DType
    {
        Float32 = 0, //float
        Float16 = 1, //Half(ushort)
        Float64 = 2, //double
        Int32   = 3, //int
        UInt8   = 4, //byte
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Half
    {
        public ushort value;
    }

    /// <summary>
    /// 
    /// </summary>
    public static class DTypeExtensions
    {
        public static int Size( this DType value )
        {
            switch ( value )
            {
                case DType.Float16: return 2; //Half(ushort)
                case DType.Float32: return 4; //float
                case DType.Float64: return 8; //double
                case DType.Int32:   return 4; //int
                case DType.UInt8:   return 1; //byte
                default:
                    throw (new NotSupportedException( $"Element type '{value}' not supported." ));
            }
        }

        //public static Type ToCLRType( this DType value )
        //{
        //    switch ( value )
        //    {
        //        case DType.Float16: return typeof(Half);
        //        case DType.Float32: return typeof(float);
        //        case DType.Float64: return typeof(double);
        //        case DType.Int32:   return typeof(int);
        //        case DType.UInt8:   return typeof(byte);
        //        default:
        //            throw (new NotSupportedException( $"Element type '{value}' not supported." ));
        //    }
        //}
    }

    /// <summary>
    /// 
    /// </summary>
    public static class DTypeBuilder
    {
        public static DType FromCLRType( Type type )
        {
            if ( type == typeof(Half)   ) return DType.Float16;
            if ( type == typeof(float)  ) return DType.Float32;
            if ( type == typeof(double) ) return DType.Float64;
            if ( type == typeof(int)    ) return DType.Int32;
            if ( type == typeof(byte)   ) return DType.UInt8;
               
            throw (new NotSupportedException( $"No corresponding DType value for CLR type '{type}'." ));
        }
    }
}

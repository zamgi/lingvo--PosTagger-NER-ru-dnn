using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace Lingvo.PosTagger.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public static class AssemblyExtensions
    {
        public static IEnumerable< (Type type, IEnumerable< T > attrs) > TypesWithAttribute< T >( this Assembly assembly, bool inherit )
        {
            foreach ( Type type in assembly.GetTypes() )
            {
                object[] attributes = type.GetCustomAttributes( typeof(T), inherit );
                if ( attributes.Any() )
                {
                    yield return (type, attributes.Cast< T >());
                }
            }
        }
    }
    /// <summary>
    /// 
    /// </summary>
    public static class TypeExtensions
    {
        public static IEnumerable< (MethodInfo methodInfo, IEnumerable< T > attrs) > MethodsWithAttribute< T >( this Type type, bool inherit )
        {
            foreach ( MethodInfo method in type.GetMethods() )
            {
                object[] attributes = method.GetCustomAttributes( typeof(T), inherit );
                if ( attributes.Any() )
                {
                    yield return (method, attributes.Cast< T >());
                }
            }
        }
    }
    /// <summary>
    /// 
    /// </summary>
    public static class MethodExtensions
    {
        public static IEnumerable< (ParameterInfo parameterInfo, IEnumerable< T > attrs) > ParametersWithAttribute< T >( this MethodInfo method, bool inherit )
        {
            foreach ( ParameterInfo paramter in method.GetParameters() )
            {
                object[] attributes = paramter.GetCustomAttributes( typeof(T), inherit );
                if ( attributes.Any() )
                {
                    yield return (paramter, attributes.Cast< T >());
                }
            }
        }
    }
}

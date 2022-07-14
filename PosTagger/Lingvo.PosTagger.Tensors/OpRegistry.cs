using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Lingvo.PosTagger.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public delegate object OpHandler( object[] args );

    /// <summary>
    /// 
    /// </summary>
    public static class OpRegistry
    {
        /// <summary>
        /// 
        /// </summary>
        private sealed class OpInstance
        {
            public OpHandler Handler;
            public IList< OpConstraint > Constraints;

            [M(O.AggressiveInlining)] public bool IsConstraintsAllSatisfiedFor( object[] args )
            {
                switch ( Constraints.Count )
                {
                    case 1: return (Constraints[ 0 ].SatisfiedFor( args ));
                    case 2: return (Constraints[ 0 ].SatisfiedFor( args ) && Constraints[ 1 ].SatisfiedFor( args ));
                    case 3: return (Constraints[ 0 ].SatisfiedFor( args ) && Constraints[ 1 ].SatisfiedFor( args ) && Constraints[ 2 ].SatisfiedFor( args ));
                    case 4: return (Constraints[ 0 ].SatisfiedFor( args ) && Constraints[ 1 ].SatisfiedFor( args ) && Constraints[ 2 ].SatisfiedFor( args ) && Constraints[ 3 ].SatisfiedFor( args ));
                    case 5: return (Constraints[ 0 ].SatisfiedFor( args ) && Constraints[ 1 ].SatisfiedFor( args ) && Constraints[ 2 ].SatisfiedFor( args ) && Constraints[ 3 ].SatisfiedFor( args ) && Constraints[ 4 ].SatisfiedFor( args ));
                    default:
                        for ( var i = Constraints.Count - 1; 0 <= i; i-- )
                        {
                            if ( !Constraints[ i ].SatisfiedFor( args ) )
                            {
                                return (false);
                            }
                        }
                        return (true);
                }
            }
        }

        private static readonly Dictionary< string, List< OpInstance > > _OpInstances;
        // Remember which assemblies have been registered to avoid accidental double-registering
        private static readonly HashSet<Assembly> _RegisteredAssemblies;
        static OpRegistry()
        {
            _OpInstances = new Dictionary<string, List<OpInstance>>();
            _RegisteredAssemblies = new HashSet<Assembly>();

            // Register CPU ops from this assembly
            RegisterAssembly( Assembly.GetExecutingAssembly() );
        }

        public static void Register( string opName, OpHandler handler, IList< OpConstraint > constraints )
        {
            var newInstance = new OpInstance() { Handler = handler, Constraints = constraints };
            if ( _OpInstances.TryGetValue( opName, out var instanceList ) )
            {
                instanceList.Add( newInstance );
            }
            else
            {
                instanceList = new List< OpInstance > { newInstance };
                _OpInstances.Add( opName, instanceList );
            }
        }

        public static object Invoke( string opName, params object[] args )
        {
            if ( _OpInstances.TryGetValue( opName, out var instanceList ) )
            {
                switch ( instanceList.Count )
                {
                    case 0: break;
                    case 1:
                        var instance_0 = instanceList[ 0 ];
                        if ( instance_0.IsConstraintsAllSatisfiedFor( args ) ) //---if ( instance.constraints.All( x => x.SatisfiedFor( args ) ) )
                        {
                            return (instance_0.Handler.Invoke( args ));
                        }
                        break;

                    default:
                        foreach ( OpInstance instance in instanceList )
                        {
                            if ( instance.IsConstraintsAllSatisfiedFor( args ) ) //---if ( instance.constraints.All( x => x.SatisfiedFor( args ) ) )
                            {
                                return (instance.Handler.Invoke( args ));
                            }
                        }
                        break;
                }

                throw (new ApplicationException( $"None of the registered handlers match the arguments for '{opName}'." ));
            }
            else
            {
                throw (new ApplicationException( $"No handlers have been registered for op '{opName}'." ));
            }
        }

        public static void RegisterAssembly( Assembly assembly )
        {
            if ( _RegisteredAssemblies.Add( assembly ) )
            {
                var types = assembly.TypesWithAttribute< OpsClassAttribute >( false ).Select( x => x.type );

                foreach ( var type in types )
                {
                    object instance = Activator.CreateInstance( type );

                    var ts = type.MethodsWithAttribute< RegisterOp >( false );
                    foreach ( var t in ts )
                    {
                        IEnumerable< OpConstraint > paramConstraints = GetParameterConstraints( t.methodInfo, instance );
                        foreach ( RegisterOp attribute in t.attrs )
                        {
                            attribute.DoRegister( instance, t.methodInfo, paramConstraints );
                        }
                    }
                }
            }
        }

        private static IEnumerable< OpConstraint > GetParameterConstraints( MethodInfo method, object instance )
        {
            var result = Enumerable.Empty< OpConstraint >();
            foreach ( var t in method.ParametersWithAttribute< ArgConstraintAttribute >( false ) )
            {
                foreach ( ArgConstraintAttribute attribute in t.attrs )
                {
                    result = Enumerable.Concat( result, attribute.GetConstraints( t.parameterInfo, instance ) );
                }
            }
            return (result);
        }
    }
}

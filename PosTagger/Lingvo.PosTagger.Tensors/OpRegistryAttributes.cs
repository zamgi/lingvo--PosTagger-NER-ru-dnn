using System;
using System.Collections.Generic;
using System.Reflection;

namespace Lingvo.PosTagger.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    [AttributeUsage( AttributeTargets.Class )]
    public class OpsClassAttribute : Attribute
    {
    }

    /// <summary>
    /// 
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public abstract class RegisterOp : Attribute
    {
        public RegisterOp( string opName ) => OpName = opName;
        public string OpName { get; }
        public abstract void DoRegister( object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints );
    }

    /// <summary>
    /// Register a method where the only constraint is that the argument counts match.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpArgCount : RegisterOp
    {
        public RegisterOpArgCount( string opName ) : base( opName ) { }
        public override void DoRegister( object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints )
        {
            var constraints = new List<OpConstraint>();
            constraints.AddRange( paramConstraints );
            constraints.Add( new ArgCountConstraint( method.GetParameters().Length ) );

            OpRegistry.Register( OpName, args => method.Invoke( instance, args ), constraints );
        }
    }

    /// <summary>
    /// 
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpStorageType : RegisterOp
    {
        private readonly Type _StorageType;
        public RegisterOpStorageType( string opName, Type storageType ) : base( opName ) => _StorageType = storageType;
        public override void DoRegister( object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints )
        {
            var constraints = new List< OpConstraint >();
            constraints.AddRange( paramConstraints );
            constraints.Add( new ArgCountConstraint( method.GetParameters().Length ) );

            ParameterInfo[] methodParams = method.GetParameters();
            for ( int i = 0; i < methodParams.Length; ++i )
            {
                if ( methodParams[ i ].ParameterType == typeof(Tensor) )
                {
                    constraints.Add( new ArgStorageTypeConstraint( i, _StorageType ) );
                }
            }
            OpRegistry.Register( OpName, args => method.Invoke( instance, args ), constraints );
        }
    }

    /// <summary>
    /// 
    /// </summary>
    [AttributeUsage(AttributeTargets.Parameter)]
    public abstract class ArgConstraintAttribute : Attribute
    {
        public ArgConstraintAttribute() { }
        public abstract IEnumerable<OpConstraint> GetConstraints( ParameterInfo parameter, object instance );
    }

    /// <summary>
    /// 
    /// </summary>
    [AttributeUsage(AttributeTargets.Parameter)]
    public class OpArgStorageType : ArgConstraintAttribute
    {
        private readonly Type _StorageType;
        public OpArgStorageType( Type storageType ) => _StorageType = storageType;
        public override IEnumerable<OpConstraint> GetConstraints( ParameterInfo parameter, object instance )
        {
            yield return (new ArgStorageTypeConstraint( parameter.Position, _StorageType ));
        }
    }
}

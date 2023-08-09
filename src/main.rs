use std::collections::BTreeMap;
use std::rc::Rc;

use rcl::ast::{BinOp, Expr, Seq, UnOp};
use rcl::runtime::{Env, Value};
use rcl::source::{DocId, Document, Inputs};

/// Helpers for constructing AST in code.
pub fn var(name: &'static str) -> Expr {
    Expr::Var(name.into())
}

pub fn string(value: &'static str) -> Expr {
    Expr::StringLit(value.into())
}

pub fn field(field: &'static str, obj: Expr) -> Expr {
    Expr::Field(field.into(), Box::new(obj))
}

pub fn singleton(key: &'static str, value: Expr) -> Vec<Seq> {
    vec![Seq::Assoc {
        key: Box::new(string(key.into())),
        value: Box::new(value),
    }]
}

pub fn neg(x: Expr) -> Expr {
    Expr::UnOp(UnOp::Neg, Box::new(x))
}

pub fn union(x: Expr, y: Expr) -> Expr {
    Expr::BinOp(BinOp::Union, Box::new(x), Box::new(y))
}

pub fn call(f: Expr, args: Vec<Expr>) -> Expr {
    Expr::Call(Box::new(f), args)
}

pub fn let_compr(name: &'static str, value: Expr, in_: Seq) -> Seq {
    Seq::Let {
        ident: name.into(),
        value: Box::new(value),
        body: Box::new(in_),
    }
}

pub fn assoc(key: Expr, value: Expr) -> Seq {
    Seq::Assoc {
        key: Box::new(key),
        value: Box::new(value),
    }
}

fn example_ast() -> Expr {
    let inner_if = Seq::If {
        condition: Box::new(call(
            field(
                "contains",
                call(
                    field("get", field("group_devices", var("var"))),
                    vec![var("group"), Expr::BraceLit(vec![])],
                ),
            ),
            vec![var("host")],
        )),
        body: Box::new(Seq::For {
            idents: vec!["tag".into()],
            collection: Box::new(var("tags")),
            body: Box::new(Seq::Elem(Box::new(var("tag")))),
        }),
    };

    let body = let_compr(
        "device_tags",
        call(
            field("get", field("device_tags", var("var"))),
            vec![var("host"), Expr::BracketLit(vec![])],
        ),
        let_compr(
            "group_tags",
            Expr::BraceLit(vec![Seq::For {
                idents: vec!["group".into(), "tags".into()],
                collection: Box::new(field("group_tags", var("var"))),
                body: Box::new(inner_if),
            }]),
            assoc(var("host"), union(var("group_tags"), var("device_tags"))),
        ),
    );

    Expr::BraceLit(singleton(
        "tags",
        Expr::BraceLit(vec![Seq::For {
            idents: vec!["host".into()],
            collection: Box::new(field("hosts", var("var"))),
            body: Box::new(Seq::If {
                condition: Box::new(neg(call(
                    field("contains", field("excluded_devices", var("var"))),
                    vec![var("host")],
                ))),
                body: Box::new(body),
            }),
        }]),
    ))
}

fn val_string(v: &str) -> Rc<Value> {
    Rc::new(Value::String(v.into()))
}

fn val_list(vs: &[&Rc<Value>]) -> Rc<Value> {
    Rc::new(Value::List(vs.iter().copied().cloned().collect()))
}

fn example_env() -> Env {
    #![allow(clippy::redundant_clone)]

    let ams01 = val_string("ams01");
    let ams02 = val_string("ams02");
    let fra01 = val_string("fra01");
    let fra02 = val_string("fra02");
    let fra03 = val_string("fra03");
    let lax01 = val_string("lax01");
    let zrh01 = val_string("zrh01");
    let zrh02 = val_string("zrh02");

    let group_ch = val_string("group:ch");
    let group_de = val_string("group:de");
    let group_nl = val_string("group:nl");
    let group_us = val_string("group:us");

    let icelake = val_string("tag:icelake");
    let skylake = val_string("tag:skylake");
    let znver2 = val_string("tag:znver2");
    let znver3 = val_string("tag:znver3");
    let primary = val_string("tag:primary");
    let standby = val_string("tag:standby");
    let expensive = val_string("tag:expensive");
    let high_latency = val_string("tag:high_latency");
    let colocated = val_string("tag:colocated");

    let hosts = val_list(&[
        &ams01, &ams02, &fra01, &fra02, &fra03, &lax01, &zrh01, &zrh02,
    ]);

    let excluded_devices = val_list(&[&lax01]);

    let mut device_tags = BTreeMap::new();
    device_tags.insert(ams01.clone(), val_list(&[&znver2, &primary]));
    device_tags.insert(ams02.clone(), val_list(&[&znver3, &standby]));
    device_tags.insert(fra01.clone(), val_list(&[&icelake]));
    device_tags.insert(fra02.clone(), val_list(&[&icelake]));
    device_tags.insert(fra03.clone(), val_list(&[&icelake]));
    device_tags.insert(lax01.clone(), val_list(&[&skylake]));
    device_tags.insert(zrh01.clone(), val_list(&[&znver3, &standby]));
    device_tags.insert(zrh02.clone(), val_list(&[&znver3, &primary]));

    let mut group_devices = BTreeMap::new();
    group_devices.insert(group_ch.clone(), val_list(&[&zrh01, &zrh02]));
    group_devices.insert(group_de.clone(), val_list(&[&fra01, &fra02, &fra03]));
    group_devices.insert(group_nl.clone(), val_list(&[&ams01, &ams02]));
    group_devices.insert(group_us.clone(), val_list(&[&lax01]));

    let mut group_tags = BTreeMap::new();
    group_tags.insert(group_ch.clone(), val_list(&[&expensive, &colocated]));
    group_tags.insert(group_nl.clone(), val_list(&[&colocated]));
    group_tags.insert(group_us.clone(), val_list(&[&high_latency]));

    let mut vars = BTreeMap::new();
    vars.insert(val_string("hosts"), hosts);
    vars.insert(val_string("excluded_devices"), excluded_devices);
    vars.insert(val_string("device_tags"), Rc::new(Value::Map(device_tags)));
    vars.insert(
        val_string("group_devices"),
        Rc::new(Value::Map(group_devices)),
    );
    vars.insert(val_string("group_tags"), Rc::new(Value::Map(group_tags)));

    let mut env = Env::new();
    env.push("var".into(), Rc::new(Value::Map(vars)));

    env
}

fn main_tags(inputs: &Inputs) -> rcl::error::Result<()> {
    for (i, doc) in inputs.iter().enumerate() {
        let id = DocId(i as u32);
        let tokens = rcl::lexer::lex(id, doc.data)?;
        for (token, span) in &tokens {
            eprintln!("{span:?} {token:?}");
        }

        let cst = rcl::parser::parse(id, doc.data)?;
        eprintln!("{cst:#?}");

        let ast = rcl::abstraction::abstract_expr(doc.data, &cst);
        eprintln!("{ast:#?}");
    }

    Ok(())
}

fn main() {
    let expr = example_ast();
    let mut env = example_env();
    eprintln!("{:#?}", expr);
    eprintln!("{:#?}", env);
    let result = rcl::eval::eval(&mut env, &expr).expect("Failed to evaluate.");
    let mut result_json = String::new();
    rcl::json::format_json(result.as_ref(), &mut result_json).expect("Failed to format json.");
    println!("{}", result_json);

    let fname = "examples/tags.rcl";
    let data = std::fs::read_to_string(fname).expect("Failed to load example.");
    let doc = Document {
        path: fname,
        data: &data,
    };
    let inputs = [doc];
    if let Err(err) = main_tags(&inputs) {
        err.print(&inputs);
    }
}
